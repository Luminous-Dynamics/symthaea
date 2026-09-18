// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::{AuthorityProfile, EvidenceRecord, ResearchId, Sha256Digest};
use serde::Serialize;
use std::collections::BTreeSet;

/// Authority-bearing claim assembled only through validated evidence. It is
/// intentionally not directly deserializable; recovery/import must re-present
/// the underlying evidence and rebuild the claim through `from_evidence`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScientificClaim {
    claim_id: ResearchId,
    subject_sha256: Sha256Digest,
    evidence_ids: BTreeSet<ResearchId>,
    authority: AuthorityProfile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimIssue {
    EmptyEvidence,
    InvalidEvidence { evidence_id: ResearchId },
    SubjectMismatch { evidence_id: ResearchId },
}

impl ScientificClaim {
    /// Build a claim only from already-bounded evidence records.
    ///
    /// Evidence count does not manufacture replication or independence. Those
    /// facets appear only when an input record explicitly carries them.
    pub fn from_evidence(
        claim_id: ResearchId,
        subject_sha256: Sha256Digest,
        evidence: &[EvidenceRecord],
    ) -> Result<Self, Vec<ClaimIssue>> {
        let mut issues = Vec::new();
        if evidence.is_empty() {
            issues.push(ClaimIssue::EmptyEvidence);
        }
        for record in evidence {
            if !record.validate().is_empty() {
                issues.push(ClaimIssue::InvalidEvidence {
                    evidence_id: record.evidence_id().clone(),
                });
            }
            if record.subject_sha256() != &subject_sha256 {
                issues.push(ClaimIssue::SubjectMismatch {
                    evidence_id: record.evidence_id().clone(),
                });
            }
        }
        if !issues.is_empty() {
            return Err(issues);
        }

        let evidence_ids = evidence
            .iter()
            .map(|record| record.evidence_id().clone())
            .collect();
        let authority = evidence.iter().fold(AuthorityProfile::empty(), |acc, record| {
            acc.evidence_union(record.authority())
        });

        Ok(Self {
            claim_id,
            subject_sha256,
            evidence_ids,
            authority,
        })
    }

    pub fn claim_id(&self) -> &ResearchId {
        &self.claim_id
    }

    pub fn subject_sha256(&self) -> &Sha256Digest {
        &self.subject_sha256
    }

    pub fn evidence_ids(&self) -> &BTreeSet<ResearchId> {
        &self.evidence_ids
    }

    pub fn authority(&self) -> &AuthorityProfile {
        &self.authority
    }

    /// Conservative compatibility adapter. The target representation may lower
    /// authority but can never raise any facet beyond this claim.
    pub fn adapt_with_ceiling(&self, ceiling: &AuthorityProfile) -> AuthorityProfile {
        self.authority.bounded_by(ceiling)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AuthorityFacet, AuthorityLevel, EvidenceKind, EvidenceState};

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }

    fn digest(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn computation(subject: &Sha256Digest, evidence_id: &str) -> EvidenceRecord {
        EvidenceRecord::new(
            id(evidence_id),
            subject.clone(),
            EvidenceKind::DeterministicComputation,
            EvidenceState::Pass,
            digest(evidence_id),
            [],
            None,
            AuthorityProfile::empty()
                .with(AuthorityFacet::Provenance, AuthorityLevel::Bound)
                .with(AuthorityFacet::Execution, AuthorityLevel::Bound),
        )
        .unwrap()
    }

    #[test]
    fn mixed_subject_evidence_fails_closed() {
        let a = digest("subject-a");
        let b = digest("subject-b");
        let evidence = vec![computation(&a, "E-A"), computation(&b, "E-B")];
        assert!(ScientificClaim::from_evidence(id("CLAIM-1"), a, &evidence).is_err());
    }

    #[test]
    fn evidence_count_does_not_invent_replication_or_independence() {
        let subject = digest("subject");
        let evidence = vec![
            computation(&subject, "E-1"),
            computation(&subject, "E-2"),
            computation(&subject, "E-3"),
        ];
        let claim = ScientificClaim::from_evidence(id("CLAIM-1"), subject, &evidence).unwrap();
        assert_eq!(claim.authority().get(AuthorityFacet::Replication), AuthorityLevel::None);
        assert_eq!(claim.authority().get(AuthorityFacet::Independence), AuthorityLevel::None);
    }

    #[test]
    fn adapter_can_only_preserve_or_lower_authority() {
        let subject = digest("subject");
        let evidence = vec![computation(&subject, "E-1")];
        let claim = ScientificClaim::from_evidence(id("CLAIM-1"), subject, &evidence).unwrap();
        let attempted_target = AuthorityProfile::empty()
            .with(AuthorityFacet::Execution, AuthorityLevel::Bound)
            .with(AuthorityFacet::Causal, AuthorityLevel::Bound);
        let adapted = claim.adapt_with_ceiling(&attempted_target);
        assert_eq!(adapted.get(AuthorityFacet::Execution), AuthorityLevel::Bound);
        assert_eq!(adapted.get(AuthorityFacet::Causal), AuthorityLevel::None);
    }
}
