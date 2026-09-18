// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification-bound claim identities.
//!
//! A [`SealedClaim`] intentionally identifies mathematical content independently
//! of who reviewed the underlying challenge. Authority-bearing execution needs a
//! stronger identity: the exact claim *under the exact qualification lineage*.
//! This module supplies that binding without contaminating stable claim identity.

use crate::{FrozenChallenge, SealedClaim, Sha256Digest};

const QUALIFIED_CLAIM_DOMAIN: &str = "symthaea.math-qualified-claim.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationBindingIssue {
    ChallengeMismatch,
}

/// A mathematical claim bound to one exact frozen-specification qualification.
///
/// Fields are private so downstream verifier-specific code cannot substitute a
/// different review lineage while retaining the same binding identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedClaim {
    claim: SealedClaim,
    challenge_sha256: Sha256Digest,
    qualification_sha256: Sha256Digest,
    binding_sha256: Sha256Digest,
}

impl QualifiedClaim {
    pub fn bind(
        claim: SealedClaim,
        challenge: &FrozenChallenge,
    ) -> Result<Self, QualificationBindingIssue> {
        if claim.challenge_sha256() != challenge.challenge_sha256() {
            return Err(QualificationBindingIssue::ChallengeMismatch);
        }
        let binding_sha256 = compute_binding_digest(
            claim.claim_sha256(),
            challenge.challenge_sha256(),
            challenge.qualification_sha256(),
        );
        Ok(Self {
            claim,
            challenge_sha256: challenge.challenge_sha256().clone(),
            qualification_sha256: challenge.qualification_sha256().clone(),
            binding_sha256,
        })
    }

    pub fn claim(&self) -> &SealedClaim {
        &self.claim
    }

    pub fn claim_sha256(&self) -> &Sha256Digest {
        self.claim.claim_sha256()
    }

    pub fn challenge_sha256(&self) -> &Sha256Digest {
        &self.challenge_sha256
    }

    pub fn qualification_sha256(&self) -> &Sha256Digest {
        &self.qualification_sha256
    }

    pub fn binding_sha256(&self) -> &Sha256Digest {
        &self.binding_sha256
    }
}

fn compute_binding_digest(
    claim_sha256: &Sha256Digest,
    challenge_sha256: &Sha256Digest,
    qualification_sha256: &Sha256Digest,
) -> Sha256Digest {
    let mut bytes = Vec::new();
    put_text(&mut bytes, QUALIFIED_CLAIM_DOMAIN);
    put_text(&mut bytes, claim_sha256.as_str());
    put_text(&mut bytes, challenge_sha256.as_str());
    put_text(&mut bytes, qualification_sha256.as_str());
    Sha256Digest::of_bytes(&bytes)
}

fn put_text(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_be_bytes());
    output.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ClaimDraft, FormalizationReview, FormalizationReviewKind, MathematicalClaimKind,
        MathematicalSpecification, SourceReference, SpecificationStatus,
    };

    fn digest(label: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(label.as_bytes())
    }

    fn qualified_challenge(independent_reviewer: &str, independent_domain: &str) -> FrozenChallenge {
        let mut spec = MathematicalSpecification::new(
            "QUAL-TEST",
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
        let statement = spec.lean_statement_sha256.clone();
        let definitions = spec.definitions_sha256.clone();
        for (reviewer, domain, kind) in [
            ("reviewer-a", "lineage-a", FormalizationReviewKind::DefinitionsAudit),
            ("reviewer-a", "lineage-a", FormalizationReviewKind::SemanticReview),
            (
                independent_reviewer,
                independent_domain,
                FormalizationReviewKind::IndependentFormalization,
            ),
        ] {
            spec.add_review(FormalizationReview {
                reviewer_id: reviewer.into(),
                independence_domain_sha256: digest(domain),
                method_sha256: digest(&format!("method:{reviewer}:{domain}:{kind:?}")),
                kind,
                lean_statement_sha256: statement.clone(),
                definitions_sha256: definitions.clone(),
                notes: "reviewed".into(),
            });
        }
        spec.advance_to(SpecificationStatus::FrozenChallenge).unwrap();
        spec.freeze().unwrap()
    }

    fn claim(challenge: &FrozenChallenge) -> SealedClaim {
        ClaimDraft::for_challenge(
            challenge,
            "claim-1",
            digest("claim-statement"),
            digest("assumptions"),
            MathematicalClaimKind::Lemma,
            "test-agent",
            digest("context"),
        )
        .seal(challenge)
        .unwrap()
    }

    #[test]
    fn same_claim_under_different_review_lineages_gets_distinct_authority_binding() {
        let left = qualified_challenge("reviewer-b", "lineage-b");
        let right = qualified_challenge("reviewer-c", "lineage-c");
        assert_eq!(left.challenge_sha256(), right.challenge_sha256());
        assert_ne!(left.qualification_sha256(), right.qualification_sha256());

        let claim = claim(&left);
        let left_bound = QualifiedClaim::bind(claim.clone(), &left).unwrap();
        let right_bound = QualifiedClaim::bind(claim, &right).unwrap();
        assert_eq!(left_bound.claim_sha256(), right_bound.claim_sha256());
        assert_ne!(left_bound.binding_sha256(), right_bound.binding_sha256());
    }

    #[test]
    fn different_subject_cannot_reuse_qualification_binding() {
        let left = qualified_challenge("reviewer-b", "lineage-b");
        let mut different = MathematicalSpecification::new(
            "OTHER",
            SourceReference {
                locator: "https://example.invalid/other".into(),
                revision: "v1".into(),
                retrieved_at_utc: "2026-09-18T00:00:00Z".into(),
            },
            digest("other-source"),
            digest("other-lean"),
            digest("other-defs"),
            "lean-test",
            "mathlib-test",
        );
        let statement = different.lean_statement_sha256.clone();
        let definitions = different.definitions_sha256.clone();
        for (reviewer, domain, kind) in [
            ("a", "lineage-a", FormalizationReviewKind::DefinitionsAudit),
            ("a", "lineage-a", FormalizationReviewKind::SemanticReview),
            ("b", "lineage-b", FormalizationReviewKind::IndependentFormalization),
        ] {
            different.add_review(FormalizationReview {
                reviewer_id: reviewer.into(),
                independence_domain_sha256: digest(domain),
                method_sha256: digest(&format!("method:{reviewer}:{domain}:{kind:?}")),
                kind,
                lean_statement_sha256: statement.clone(),
                definitions_sha256: definitions.clone(),
                notes: "reviewed".into(),
            });
        }
        different.advance_to(SpecificationStatus::FrozenChallenge).unwrap();
        let different = different.freeze().unwrap();
        let claim = claim(&left);
        assert_eq!(
            QualifiedClaim::bind(claim, &different),
            Err(QualificationBindingIssue::ChallengeMismatch)
        );
    }
}
