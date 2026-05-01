// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Knowledge shared types — mirrors mycelix-knowledge zome entries.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Claim {
    pub id: String,
    pub author_did: String,
    pub statement: String,
    pub domain: String,
    pub evidence: Vec<Evidence>,
    pub credibility: f64,
    pub status: ClaimStatus,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ClaimStatus {
    Proposed,
    UnderReview,
    Verified,
    Disputed,
    Retracted,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Evidence {
    pub source: String,
    pub evidence_type: EvidenceType,
    pub strength: f64,
    pub url: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum EvidenceType {
    Empirical,
    Statistical,
    Expert,
    Anecdotal,
    Computational,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: KnowledgeNodeType,
    pub credibility: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum KnowledgeNodeType {
    Claim,
    Source,
    Agent,
    Domain,
    Concept,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source_id: String,
    pub target_id: String,
    pub edge_type: KnowledgeEdgeType,
    pub weight: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum KnowledgeEdgeType {
    Supports,
    Contradicts,
    Cites,
    AuthoredBy,
    BelongsTo,
    RelatedTo,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactCheckVerdict {
    pub claim_id: String,
    pub verdict: Verdict,
    pub confidence: f64,
    pub reasoning: String,
    pub checked_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Verdict {
    True,
    MostlyTrue,
    Mixed,
    MostlyFalse,
    False,
    Unverifiable,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionMarket {
    pub id: String,
    pub question: String,
    pub resolution_date: i64,
    pub yes_probability: f64,
    pub volume: u64,
    pub resolved: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn claim_roundtrip() {
        let c = Claim {
            id: "c-1".into(),
            author_did: "did:t".into(),
            statement: "test claim".into(),
            domain: "science".into(),
            evidence: vec![],
            credibility: 0.8,
            status: ClaimStatus::Verified,
            created_at: 0,
        };
        let bytes = rmp_serde::to_vec_named(&c).unwrap();
        let decoded: Claim = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.status, ClaimStatus::Verified);
    }
}
