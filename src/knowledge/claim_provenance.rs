// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed provenance and explicit inference relations for source-grounded knowledge.
//!
//! HDC similarity is a candidate-retrieval signal only. These types make claim
//! identity and the evidence needed for corroboration/contradiction explicit.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Epistemic lane for the claim, kept distinct from how likely it sounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimKind {
    SourceAttested,
    TraditionInternal,
    ScholarlyInterpretation,
    ComparativeHypothesis,
    Empirical,
}

/// Explicit polarity prevents string-level negation heuristics from adjudicating claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimPolarity {
    Positive,
    Negative,
    Unknown,
}

/// Source-witness identity envelope for a single proposition or interpretation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimContext {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub witness_id: String,
    pub source_family_id: String,
    pub context_id: String,
    pub temporal_scope: Option<String>,
    /// Numeric chronology when known (e.g. an event year), not ingestion time.
    pub event_time: Option<i64>,
    pub claim_kind: ClaimKind,
    pub polarity: ClaimPolarity,
    /// Stable evidence identifiers; these are references, not copied source content.
    pub evidence_ids: Vec<String>,
}

impl ClaimContext {
    /// Fail closed when a grounded envelope is incomplete or ambiguous.
    pub fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("subject", self.subject.as_str()),
            ("predicate", self.predicate.as_str()),
            ("object", self.object.as_str()),
            ("witness_id", self.witness_id.as_str()),
            ("source_family_id", self.source_family_id.as_str()),
            ("context_id", self.context_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(format!("claim context requires non-empty {field}"));
            }
        }
        if self.evidence_ids.is_empty() {
            return Err("claim context requires at least one evidence_id".into());
        }
        if self.evidence_ids.iter().any(|id| id.trim().is_empty()) {
            return Err("claim context evidence_ids must be non-empty".into());
        }
        let unique: HashSet<&str> = self.evidence_ids.iter().map(String::as_str).collect();
        if unique.len() != self.evidence_ids.len() {
            return Err("claim context repeats evidence_ids".into());
        }
        Ok(())
    }

    /// Same proposition under the same context/time and epistemic lane.
    pub fn same_claim(&self, other: &Self) -> bool {
        normalize(&self.subject) == normalize(&other.subject)
            && normalize(&self.predicate) == normalize(&other.predicate)
            && normalize(&self.object) == normalize(&other.object)
            && self.context_id == other.context_id
            && self.temporal_scope == other.temporal_scope
            && self.event_time == other.event_time
            && self.claim_kind == other.claim_kind
            && self.polarity == other.polarity
    }

    /// Same predicate scope; object or polarity may differ and require conflict review.
    pub fn same_conflict_scope(&self, other: &Self) -> bool {
        normalize(&self.subject) == normalize(&other.subject)
            && normalize(&self.predicate) == normalize(&other.predicate)
            && self.context_id == other.context_id
            && self.temporal_scope == other.temporal_scope
            && self.event_time == other.event_time
            && self.claim_kind == other.claim_kind
    }

    /// Independence is provenance-based, never inferred from vector similarity.
    pub fn admissibly_independent_from(&self, other: &Self) -> bool {
        self.witness_id != other.witness_id
            && self.source_family_id != other.source_family_id
            && !self.evidence_ids.iter().any(|left| other.evidence_ids.contains(left))
    }
}

fn normalize(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// Relations between facts are different from the semantic similarity used to retrieve them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceRelation {
    SemanticNeighbor,
    SameWitnessVariant,
    Corroborates,
    Contradicts,
    Resembles,
    Precedes,
    Causes,
    HistoricallyInfluences,
}

/// Candidate relations must never change confidence before explicit adjudication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceLinkStatus {
    Candidate,
    Adjudicated,
    Rejected,
}

/// An explicit graph edge between claims, with audit-friendly reasons and evidence IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceLink {
    pub from_fact_id: u64,
    pub to_fact_id: u64,
    pub relation: EvidenceRelation,
    pub status: EvidenceLinkStatus,
    pub evidence_ids: Vec<String>,
    pub rationale: Option<String>,
    pub causal_argument: Option<String>,
    pub transmission_chain: Option<String>,
    pub detected_at_cycle: u64,
}

impl EvidenceLink {
    pub fn is_adjudicated(&self) -> bool {
        self.status == EvidenceLinkStatus::Adjudicated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(witness: &str, family: &str, evidence: &str) -> ClaimContext {
        ClaimContext {
            subject: "Osiris".into(),
            predicate: "associated_with".into(),
            object: "renewal".into(),
            witness_id: witness.into(),
            source_family_id: family.into(),
            context_id: "Dendera:late-period".into(),
            temporal_scope: Some("late_period".into()),
            event_time: Some(-300),
            claim_kind: ClaimKind::SourceAttested,
            polarity: ClaimPolarity::Positive,
            evidence_ids: vec![evidence.into()],
        }
    }

    #[test]
    fn claim_identity_is_not_vector_similarity() {
        let left = context("witness-a", "family-a", "evidence-a");
        let same_claim_other_witness = context("witness-b", "family-b", "evidence-b");
        let other_context = ClaimContext {
            context_id: "Abydos:middle-period".into(),
            ..same_claim_other_witness.clone()
        };

        assert!(left.same_claim(&same_claim_other_witness));
        assert!(left.admissibly_independent_from(&same_claim_other_witness));
        assert!(!left.same_claim(&other_context));
    }

    #[test]
    fn claim_context_rejects_duplicate_or_missing_evidence_ids() {
        let mut claim = context("witness-a", "family-a", "evidence-a");
        claim.evidence_ids.push("evidence-a".into());
        assert!(claim.validate().unwrap_err().contains("repeats evidence_ids"));
        claim.evidence_ids.clear();
        assert!(claim.validate().unwrap_err().contains("at least one evidence_id"));
    }

    #[test]
    fn evidence_link_candidate_is_not_adjudicated() {
        let link = EvidenceLink {
            from_fact_id: 1,
            to_fact_id: 2,
            relation: EvidenceRelation::Corroborates,
            status: EvidenceLinkStatus::Candidate,
            evidence_ids: Vec::new(),
            rationale: None,
            causal_argument: None,
            transmission_chain: None,
            detected_at_cycle: 10,
        };
        assert!(!link.is_adjudicated());
    }
}
