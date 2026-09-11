// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing assumptions and claims for infrastructure trade studies.
//!
//! The registry makes projections, requirements, observations, and assumptions
//! explicit inputs rather than allowing them to disappear into model code.

use serde::{Deserialize, Serialize};

use crate::{transport::BoundedMetric, EvidenceStatus};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClaimId(pub String);

impl ClaimId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn is_well_formed(&self) -> bool {
        !self.0.trim().is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClaimKind {
    Assumption,
    Requirement,
    Projection,
    Observation,
    Derived,
    ExternalClaim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClaimDisposition {
    Active,
    Challenged,
    Superseded,
    Rejected,
}

/// One explicit claim used by an architecture or trade study.
///
/// `scalar` is optional because some claims are qualitative requirements or
/// architecture assertions rather than numerical values. Tested/qualified
/// claims must cite evidence; a declared assumption may intentionally have no
/// external evidence yet, but remains visible as an assumption.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimRecord {
    pub claim_id: ClaimId,
    pub kind: ClaimKind,
    pub subject: String,
    pub statement: String,
    pub scalar: Option<BoundedMetric>,
    pub evidence_status: EvidenceStatus,
    pub disposition: ClaimDisposition,
    pub source_refs: Vec<String>,
    pub supersedes: Vec<ClaimId>,
}

impl ClaimRecord {
    pub fn is_well_formed(&self) -> bool {
        self.claim_id.is_well_formed()
            && !self.subject.trim().is_empty()
            && !self.statement.trim().is_empty()
            && self
                .scalar
                .as_ref()
                .is_none_or(BoundedMetric::is_well_formed)
            && self.supersedes.iter().all(ClaimId::is_well_formed)
            && !self.supersedes.iter().any(|id| id == &self.claim_id)
            && match self.evidence_status {
                EvidenceStatus::Tested | EvidenceStatus::Qualified => {
                    !self.source_refs.is_empty()
                        && self.source_refs.iter().all(|reference| !reference.trim().is_empty())
                }
                EvidenceStatus::Declared | EvidenceStatus::Modeled => self
                    .source_refs
                    .iter()
                    .all(|reference| !reference.trim().is_empty()),
            }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ClaimRegistry {
    pub claims: Vec<ClaimRecord>,
}

impl ClaimRegistry {
    pub fn is_well_formed(&self) -> bool {
        if self.claims.iter().any(|claim| !claim.is_well_formed()) {
            return false;
        }

        for (index, claim) in self.claims.iter().enumerate() {
            if self.claims[index + 1..]
                .iter()
                .any(|other| other.claim_id == claim.claim_id)
            {
                return false;
            }
        }

        self.claims.iter().all(|claim| {
            claim.supersedes.iter().all(|superseded| {
                self.claims
                    .iter()
                    .any(|candidate| &candidate.claim_id == superseded)
            })
        })
    }

    pub fn get(&self, id: &ClaimId) -> Option<&ClaimRecord> {
        self.claims.iter().find(|claim| &claim.claim_id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declared(id: &str) -> ClaimRecord {
        ClaimRecord {
            claim_id: ClaimId::new(id),
            kind: ClaimKind::Assumption,
            subject: "reference-demand".into(),
            statement: "placeholder scenario assumption".into(),
            scalar: None,
            evidence_status: EvidenceStatus::Declared,
            disposition: ClaimDisposition::Active,
            source_refs: vec![],
            supersedes: vec![],
        }
    }

    #[test]
    fn declared_assumption_can_be_explicitly_unsupported() {
        assert!(declared("a-1").is_well_formed());
    }

    #[test]
    fn tested_claim_requires_evidence_reference() {
        let mut claim = declared("test-1");
        claim.evidence_status = EvidenceStatus::Tested;
        assert!(!claim.is_well_formed());
        claim.source_refs.push("coupon-test-42".into());
        assert!(claim.is_well_formed());
    }

    #[test]
    fn registry_rejects_duplicate_ids() {
        let registry = ClaimRegistry {
            claims: vec![declared("same"), declared("same")],
        };
        assert!(!registry.is_well_formed());
    }

    #[test]
    fn superseded_claim_must_exist_in_registry() {
        let mut replacement = declared("new");
        replacement.supersedes.push(ClaimId::new("old"));
        let registry = ClaimRegistry {
            claims: vec![replacement],
        };
        assert!(!registry.is_well_formed());

        let mut old = declared("old");
        old.disposition = ClaimDisposition::Superseded;
        let mut replacement = declared("new");
        replacement.supersedes.push(ClaimId::new("old"));
        let registry = ClaimRegistry {
            claims: vec![old, replacement],
        };
        assert!(registry.is_well_formed());
    }
}
