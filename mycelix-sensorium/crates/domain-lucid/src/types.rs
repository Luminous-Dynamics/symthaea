// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! LUCID shared types — mirrors lucid zome entries.
//! Replaces 566 lines of lucid/types.ts in the TS SDK.

use serde::{Deserialize, Serialize};

// ── Thoughts ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Thought {
    pub id: String,
    pub content: String,
    pub thought_type: ThoughtType,
    pub confidence: f64,
    pub tags: Vec<String>,
    pub domain: Option<String>,
    pub epistemic: EpistemicProfile,
    pub author_did: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ThoughtType { Observation, Hypothesis, Claim, Question, Reflection, Synthesis, Critique }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpistemicProfile {
    pub empirical: f64,
    pub normative: f64,
    pub materiality: f64,
    pub harmonic: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateThoughtInput {
    pub content: String,
    pub thought_type: ThoughtType,
    pub confidence: f64,
    pub tags: Vec<String>,
    pub domain: Option<String>,
    pub epistemic: EpistemicProfile,
}

// ── Relationships ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Relationship {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relationship_type: RelationshipType,
    pub strength: f64,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RelationshipType {
    Supports, Contradicts, Extends, Refines, Inspires,
    QuestionsAssumptionOf, Synthesizes, Contextualizes,
}

// ── Collective Sensemaking ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsensusView {
    pub topic: String,
    pub participants: u32,
    pub agreement_level: f64,
    pub key_insights: Vec<String>,
    pub open_questions: Vec<String>,
    pub last_activity: i64,
}

// ── Knowledge Graph Stats ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KnowledgeGraphStats {
    pub total_thoughts: u64,
    pub total_relationships: u64,
    pub total_domains: u32,
    pub avg_confidence: f64,
    pub most_active_domain: Option<String>,
    pub coherence_score: f64,
}

// ── Reasoning ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReasoningChain {
    pub id: String,
    pub premise_ids: Vec<String>,
    pub conclusion_id: String,
    pub reasoning_type: ReasoningType,
    pub validity_score: f64,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ReasoningType { Deductive, Inductive, Abductive, Analogical, Causal }

// ── Temporal Consciousness ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsciousnessSnapshot {
    pub phi: f64,
    pub coherence: f64,
    pub integration: f64,
    pub complexity: f64,
    pub timestamp: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn thought_roundtrip() {
        let t = Thought { id: "t-1".into(), content: "test".into(), thought_type: ThoughtType::Claim, confidence: 0.8, tags: vec!["a".into()], domain: Some("phil".into()), epistemic: EpistemicProfile { empirical: 0.5, normative: 0.3, materiality: 0.5, harmonic: 0.5 }, author_did: "did:t".into(), created_at: 0, updated_at: 0 };
        let bytes = rmp_serde::to_vec_named(&t).unwrap();
        let decoded: Thought = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.thought_type, ThoughtType::Claim);
    }
}
