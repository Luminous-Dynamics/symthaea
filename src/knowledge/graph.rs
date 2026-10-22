// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Enhanced Knowledge Graph with Temporal Indexing & Contradiction Detection
//!
//! Extends the web_research KnowledgeGraph with:
//! - **Temporal indexing**: facts have timestamps, can query by time range
//! - **Confidence decay**: facts lose confidence over time unless refreshed
//! - **Contradiction detection**: new facts that contradict existing ones trigger alerts
//! - **HDC similarity search**: find facts by hypervector proximity
//!
//! Science: Temporal knowledge graphs (Trivedi et al. 2017),
//!          Belief revision (AGM theory, Alchourrón et al. 1985)

use super::claim_provenance::{
    ClaimContext, EvidenceLink, EvidenceLinkStatus, EvidenceRelation,
};
use super::encoding::FactEncoding;
use std::collections::{HashMap, HashSet};
use symthaea_core::hdc::unified_hv::BinaryHV;

// ── Types ──────────────────────────────────────────────────────────────────

/// Unique fact identifier
pub type FactId = u64;

/// A fact stored in the knowledge graph with temporal metadata
#[derive(Debug, Clone)]
pub struct TemporalFact {
    /// Unique identifier
    pub id: FactId,
    /// HDC encoding of this fact
    pub encoding: FactEncoding,
    /// Cycle when this fact was inserted
    pub inserted_at_cycle: u64,
    /// Cycle when this fact was last accessed/refreshed
    pub last_accessed_cycle: u64,
    /// Current confidence (decays over time)
    pub confidence: f32,
    /// Initial confidence at insertion
    pub initial_confidence: f32,
    /// Number of times this fact has been corroborated
    pub corroboration_count: u32,
    /// Number of times this fact has been contradicted
    pub contradiction_count: u32,
    /// Domain tag for cross-domain linking
    pub domain: Option<String>,
    /// Whether this fact has causal relations (eligible for DAG)
    pub has_causal_relations: bool,
}

/// Alert raised when a new fact contradicts an existing one
#[derive(Debug, Clone)]
pub struct ContradictionAlert {
    /// The new fact that caused the contradiction
    pub new_fact_id: FactId,
    /// The existing fact being contradicted
    pub existing_fact_id: FactId,
    /// Similarity score (higher = more directly contradictory)
    pub similarity: f32,
    /// Which fact has higher confidence
    pub higher_confidence_id: FactId,
    /// Cycle when detected
    pub detected_at_cycle: u64,
}

/// Search result from the knowledge graph
#[derive(Debug, Clone)]
pub struct FactSearchResult {
    /// The matching fact
    pub fact_id: FactId,
    /// Similarity to query
    pub similarity: f32,
    /// Current confidence
    pub confidence: f32,
}

// ── Enhanced Knowledge Graph ───────────────────────────────────────────────

/// Knowledge graph with temporal awareness and HDC similarity search
pub struct EnhancedKnowledgeGraph {
    /// All facts indexed by ID
    facts: HashMap<FactId, TemporalFact>,
    /// Next available fact ID
    next_id: FactId,
    /// Maximum number of facts before eviction
    capacity: usize,
    /// Confidence decay rate per cycle (multiplicative)
    /// Science: Ebbinghaus forgetting curve (1885), exponential decay model
    confidence_decay_rate: f32,
    /// Minimum confidence before a fact is eligible for eviction
    eviction_threshold: f32,
    /// Contradiction similarity threshold (facts this similar in opposite roles = contradiction)
    contradiction_threshold: f32,
    /// Domain index: domain_tag → Vec<FactId>
    domain_index: HashMap<String, Vec<FactId>>,
    /// Pending contradiction candidates (drained by the knowledge manager each cycle)
    pending_contradictions: Vec<ContradictionAlert>,
    /// Source-grounded claim context indexed separately to preserve TemporalFact's public shape.
    claim_contexts: HashMap<FactId, ClaimContext>,
    /// Explicit and candidate typed links between facts.
    evidence_links: Vec<EvidenceLink>,
    /// Contradiction alert identities already resolved, for idempotent resolution.
    resolved_contradictions: HashSet<(FactId, FactId, Vec<String>)>,
    /// Statistics
    total_insertions: u64,
    total_evictions: u64,
    total_contradictions: u64,
}

impl Default for EnhancedKnowledgeGraph {
    fn default() -> Self {
        Self::new(10_000)
    }
}

impl EnhancedKnowledgeGraph {
    pub fn new(capacity: usize) -> Self {
        Self {
            facts: HashMap::new(),
            next_id: 1,
            capacity,
            confidence_decay_rate: 0.9999, // Very slow decay per cycle (~7% per 1000 cycles)
            eviction_threshold: 0.05,
            contradiction_threshold: 0.7,
            domain_index: HashMap::new(),
            pending_contradictions: Vec::new(),
            claim_contexts: HashMap::new(),
            evidence_links: Vec::new(),
            resolved_contradictions: HashSet::new(),
            total_insertions: 0,
            total_evictions: 0,
            total_contradictions: 0,
        }
    }

    /// Insert a new fact into the knowledge graph.
    ///
    /// Returns the fact ID and any contradiction alerts generated.
    /// Insert a generic fact without inferring identity from HDC similarity.
    ///
    /// Similarity remains a retrieval/candidate-link signal; it cannot merge facts
    /// or increment confidence without explicit, typed evidence.
    pub fn insert(
        &mut self,
        encoding: FactEncoding,
        current_cycle: u64,
        domain: Option<String>,
        has_causal: bool,
    ) -> (FactId, Vec<ContradictionAlert>) {
        self.insert_with_context(encoding, current_cycle, domain, has_causal, None)
    }

    /// Insert a source-grounded fact with explicit claim identity.
    pub fn insert_grounded(
        &mut self,
        encoding: FactEncoding,
        current_cycle: u64,
        domain: Option<String>,
        has_causal: bool,
        claim_context: ClaimContext,
    ) -> Result<(FactId, Vec<ContradictionAlert>), String> {
        claim_context.validate()?;
        Ok(self.insert_with_context(
            encoding,
            current_cycle,
            domain,
            has_causal,
            Some(claim_context),
        ))
    }

    fn insert_with_context(
        &mut self,
        encoding: FactEncoding,
        current_cycle: u64,
        domain: Option<String>,
        has_causal: bool,
        claim_context: Option<ClaimContext>,
    ) -> (FactId, Vec<ContradictionAlert>) {
        // Similarity/text checks create candidates, never an adjudicated contradiction.
        let contradictions = self.detect_contradictions(&encoding, current_cycle);
        let similar_existing: Vec<(FactId, f32)> = self
            .facts
            .values()
            .filter_map(|fact| {
                let similarity = encoding.vector.similarity(&fact.encoding.vector);
                (similarity > 0.85).then_some((fact.id, similarity))
            })
            .collect();

        if self.facts.len() >= self.capacity {
            self.evict_lowest_confidence();
        }

        let id = self.next_id;
        self.next_id += 1;
        let confidence = encoding.confidence;
        let context_for_index = claim_context.clone();
        let fact = TemporalFact {
            id,
            encoding,
            inserted_at_cycle: current_cycle,
            last_accessed_cycle: current_cycle,
            confidence,
            initial_confidence: confidence,
            corroboration_count: 0,
            contradiction_count: 0,
            domain: domain.clone(),
            has_causal_relations: has_causal,
        };
        self.facts.insert(id, fact);
        if let Some(context) = context_for_index {
            self.claim_contexts.insert(id, context);
        }
        self.total_insertions += 1;

        if let Some(ref d) = domain {
            self.domain_index.entry(d.clone()).or_default().push(id);
        }

        // Keep semantically close facts distinct. Record only a candidate link.
        for (existing_id, similarity) in similar_existing {
            if !self.facts.contains_key(&existing_id) {
                continue;
            }
            let (relation, evidence_ids) = {
                candidate_relation(
                    self.claim_contexts.get(&id),
                    self.claim_contexts.get(&existing_id),
                )
            };
            self.evidence_links.push(EvidenceLink {
                from_fact_id: existing_id,
                to_fact_id: id,
                relation,
                status: EvidenceLinkStatus::Candidate,
                evidence_ids,
                rationale: Some(format!(
                    "HDC similarity {similarity:.4} generated a candidate only; no inference or confidence update"
                )),
                causal_argument: None,
                transmission_chain: None,
                detected_at_cycle: current_cycle,
            });
        }

        self.pending_contradictions.extend(contradictions.clone());
        (id, contradictions)
    }

    /// Typed semantic/evidence links. Candidate links never change confidence.
    pub fn evidence_links(&self) -> &[EvidenceLink] {
        &self.evidence_links
    }

    fn validate_imported_evidence_link(&self, link: &EvidenceLink) -> Result<(), String> {
        if link.from_fact_id == link.to_fact_id {
            return Err("evidence links must connect two distinct facts".into());
        }
        let from = self
            .facts
            .get(&link.from_fact_id)
            .ok_or_else(|| format!("evidence link source {} is missing", link.from_fact_id))?;
        let to = self
            .facts
            .get(&link.to_fact_id)
            .ok_or_else(|| format!("evidence link target {} is missing", link.to_fact_id))?;

        if link.status == EvidenceLinkStatus::Rejected
            && link.rationale.as_deref().is_none_or(str::is_empty)
        {
            return Err("rejected evidence links require a rationale".into());
        }
        if link.status != EvidenceLinkStatus::Adjudicated {
            return Ok(());
        }

        let supplied: HashSet<&str> = link.evidence_ids.iter().map(String::as_str).collect();
        if supplied.is_empty()
            || supplied.len() != link.evidence_ids.len()
            || link.evidence_ids.iter().any(|id| id.trim().is_empty())
            || link.rationale.as_deref().is_none_or(str::is_empty)
        {
            return Err("adjudicated evidence links require a rationale and unique evidence IDs".into());
        }
        let from_context = from
            .claim_context
            .as_ref()
            .ok_or_else(|| format!("adjudicated link source {} has no claim context", link.from_fact_id))?;
        let to_context = to
            .claim_context
            .as_ref()
            .ok_or_else(|| format!("adjudicated link target {} has no claim context", link.to_fact_id))?;
        if !from_context.evidence_ids.iter().any(|id| supplied.contains(id.as_str()))
            || !to_context.evidence_ids.iter().any(|id| supplied.contains(id.as_str()))
        {
            return Err("adjudicated link evidence must reference both claim contexts".into());
        }
        let context_evidence: HashSet<&str> = from_context
            .evidence_ids
            .iter()
            .chain(to_context.evidence_ids.iter())
            .map(String::as_str)
            .collect();
        if !supplied.is_subset(&context_evidence) {
            return Err(
                "adjudicated link evidence contains IDs not declared by either claim context"
                    .into(),
            );
        }
        match link.relation {
            EvidenceRelation::SemanticNeighbor | EvidenceRelation::SameWitnessVariant => {
                return Err("candidate-only relations cannot be adjudicated".into());
            }
            EvidenceRelation::Corroborates => {
                if !from_context.same_claim(to_context)
                    || !from_context.admissibly_independent_from(to_context)
                {
                    return Err("adjudicated corroboration violates claim identity or source independence".into());
                }
            }
            EvidenceRelation::Contradicts => {
                if !from_context.same_conflict_scope(to_context)
                    || from_context.same_claim(to_context)
                {
                    return Err("adjudicated contradiction violates scoped-claim invariants".into());
                }
            }
            EvidenceRelation::Precedes => {
                if !strictly_precedes(from_context, to_context) {
                    return Err("adjudicated precedes relation requires explicit increasing event times".into());
                }
            }
            EvidenceRelation::Causes => {
                if !strictly_precedes(from_context, to_context)
                    || link.causal_argument.as_deref().is_none_or(str::is_empty)
                {
                    return Err("adjudicated causes relation requires temporal precedence and a causal argument".into());
                }
            }
            EvidenceRelation::HistoricallyInfluences => {
                if !strictly_precedes(from_context, to_context)
                    || link.transmission_chain.as_deref().is_none_or(str::is_empty)
                {
                    return Err("adjudicated historical influence requires temporal precedence and a transmission chain".into());
                }
            }
            EvidenceRelation::Resembles => {}
        }
        Ok(())
    }

    /// Import persisted links only after validating all endpoint, evidence and
    /// relation invariants. No partial imports occur on validation failure.
    pub fn import_evidence_links(&mut self, links: &[EvidenceLink]) -> Result<usize, String> {
        let mut unique_incoming: Vec<EvidenceLink> = Vec::new();
        for link in links {
            self.validate_imported_evidence_link(link)?;
            if unique_incoming.contains(link) || self.evidence_links.contains(link) {
                continue;
            }
            let symmetric = matches!(
                link.relation,
                EvidenceRelation::Corroborates
                    | EvidenceRelation::Contradicts
                    | EvidenceRelation::Resembles
                    | EvidenceRelation::SemanticNeighbor
                    | EvidenceRelation::SameWitnessVariant
            );
            let mut already_adjudicated = false;
            for existing in self.evidence_links.iter().chain(unique_incoming.iter()) {
                let same_direction = existing.from_fact_id == link.from_fact_id
                    && existing.to_fact_id == link.to_fact_id;
                let reverse_direction = symmetric
                    && existing.from_fact_id == link.to_fact_id
                    && existing.to_fact_id == link.from_fact_id;
                if (same_direction || reverse_direction)
                    && existing.relation == link.relation
                    && existing.status == EvidenceLinkStatus::Adjudicated
                    && link.status == EvidenceLinkStatus::Adjudicated
                {
                    if existing.evidence_ids == link.evidence_ids {
                        // Idempotent re-import of the same adjudication, even if
                        // the persisted row differs only by symmetric orientation.
                        already_adjudicated = true;
                        break;
                    }
                    return Err(
                        "fact pair/relation already has an adjudication with different evidence"
                            .into(),
                    );
                }
            }
            if !already_adjudicated {
                unique_incoming.push(link.clone());
            }
        }

        // Adjudication finalizes a prior candidate for the same pair/relation.
        let finalized: Vec<(FactId, FactId, EvidenceRelation)> = unique_incoming
            .iter()
            .filter(|link| link.status == EvidenceLinkStatus::Adjudicated)
            .map(|link| (link.from_fact_id, link.to_fact_id, link.relation))
            .collect();
        let is_symmetric = |relation: EvidenceRelation| {
            matches!(
                relation,
                EvidenceRelation::Corroborates
                    | EvidenceRelation::Contradicts
                    | EvidenceRelation::Resembles
                    | EvidenceRelation::SemanticNeighbor
                    | EvidenceRelation::SameWitnessVariant
            )
        };
        unique_incoming.retain(|link| {
            if link.status != EvidenceLinkStatus::Candidate {
                return true;
            }
            !finalized.iter().any(|(from, to, relation)| {
                *relation == link.relation
                    && ((*from == link.from_fact_id && *to == link.to_fact_id)
                        || (is_symmetric(*relation)
                            && *from == link.to_fact_id
                            && *to == link.from_fact_id))
            })
        });
        self.evidence_links.retain(|existing| {
            if existing.status != EvidenceLinkStatus::Candidate {
                return true;
            }
            !finalized.iter().any(|(from, to, relation)| {
                *relation == existing.relation
                    && ((*from == existing.from_fact_id && *to == existing.to_fact_id)
                        || (is_symmetric(*relation)
                            && *from == existing.to_fact_id
                            && *to == existing.from_fact_id))
            })
        });

        for link in &unique_incoming {
            if link.status == EvidenceLinkStatus::Adjudicated {
                match link.relation {
                    EvidenceRelation::Corroborates => {
                        for fact_id in [link.from_fact_id, link.to_fact_id] {
                            if let Some(fact) = self.facts.get_mut(&fact_id) {
                                fact.corroboration_count = fact.corroboration_count.saturating_add(1);
                            }
                        }
                    }
                    EvidenceRelation::Contradicts => {
                        for fact_id in [link.from_fact_id, link.to_fact_id] {
                            if let Some(fact) = self.facts.get_mut(&fact_id) {
                                fact.contradiction_count = fact.contradiction_count.saturating_add(1);
                            }
                        }
                        self.total_contradictions = self.total_contradictions.saturating_add(1);
                    }
                    _ => {}
                }
            }
        }
        let count = unique_incoming.len();
        self.evidence_links.extend(unique_incoming);
        Ok(count)
    }

    /// Reject a generated candidate without changing either fact's confidence.
    pub fn reject_evidence_candidate(
        &mut self,
        from_fact_id: FactId,
        to_fact_id: FactId,
        relation: EvidenceRelation,
        rationale: String,
        current_cycle: u64,
    ) -> Result<EvidenceLink, String> {
        if rationale.trim().is_empty() {
            return Err("rejecting a candidate requires a rationale".into());
        }
        let index = self
            .evidence_links
            .iter()
            .position(|link| {
                link.from_fact_id == from_fact_id
                    && link.to_fact_id == to_fact_id
                    && link.relation == relation
                    && link.status == EvidenceLinkStatus::Candidate
            })
            .ok_or_else(|| "matching candidate evidence link not found".to_string())?;
        let mut link = self.evidence_links[index].clone();
        link.status = EvidenceLinkStatus::Rejected;
        link.rationale = Some(rationale);
        link.detected_at_cycle = current_cycle;
        self.evidence_links[index] = link.clone();
        Ok(link)
    }

    /// Adjudicate a typed relation using source evidence referenced by both claims.
    #[allow(clippy::too_many_arguments)]
    pub fn adjudicate_evidence_relation(
        &mut self,
        from_fact_id: FactId,
        to_fact_id: FactId,
        relation: EvidenceRelation,
        evidence_ids: Vec<String>,
        rationale: String,
        causal_argument: Option<String>,
        transmission_chain: Option<String>,
        current_cycle: u64,
    ) -> Result<EvidenceLink, String> {
        if from_fact_id == to_fact_id {
            return Err("an evidence relation requires two distinct facts".into());
        }
        if rationale.trim().is_empty() {
            return Err("adjudicated evidence relations require a rationale".into());
        }
        let mut evidence_ids = evidence_ids;
        evidence_ids.sort();
        let supplied: HashSet<&str> = evidence_ids.iter().map(String::as_str).collect();
        if evidence_ids.is_empty()
            || evidence_ids.iter().any(|id| id.trim().is_empty())
            || supplied.len() != evidence_ids.len()
        {
            return Err("adjudication requires unique, non-empty evidence IDs".into());
        }

        let from_context = self
            .claim_contexts
            .get(&from_fact_id)
            .ok_or_else(|| format!("fact {from_fact_id} has no claim provenance"))?;
        let to_context = self
            .claim_contexts
            .get(&to_fact_id)
            .ok_or_else(|| format!("fact {to_fact_id} has no claim provenance"))?;
        let from_has_evidence = from_context
            .evidence_ids
            .iter()
            .any(|id| supplied.contains(id.as_str()));
        let to_has_evidence = to_context
            .evidence_ids
            .iter()
            .any(|id| supplied.contains(id.as_str()));
        if !from_has_evidence || !to_has_evidence {
            return Err("adjudication evidence must reference both claim contexts".into());
        }
        let context_evidence: HashSet<&str> = from_context
            .evidence_ids
            .iter()
            .chain(to_context.evidence_ids.iter())
            .map(String::as_str)
            .collect();
        if !supplied.is_subset(&context_evidence) {
            return Err(
                "adjudication evidence contains IDs not declared by either claim context".into(),
            );
        }

        match relation {
            EvidenceRelation::SemanticNeighbor | EvidenceRelation::SameWitnessVariant => {
                return Err("semantic proximity and witness variants are candidate-only relations".into());
            }
            EvidenceRelation::Corroborates => {
                if !from_context.same_claim(to_context)
                    || !from_context.admissibly_independent_from(to_context)
                {
                    return Err("corroboration requires the same scoped claim and independent source evidence".into());
                }
            }
            EvidenceRelation::Contradicts => {
                if !from_context.same_conflict_scope(to_context)
                    || from_context.same_claim(to_context)
                {
                    return Err("contradiction requires incompatible claims in the same subject/relation/context/time scope".into());
                }
            }
            EvidenceRelation::Resembles => {}
            EvidenceRelation::Precedes => {
                if !strictly_precedes(from_context, to_context) {
                    return Err("precedes requires explicit event times with from earlier than to".into());
                }
            }
            EvidenceRelation::Causes => {
                if !strictly_precedes(from_context, to_context)
                    || causal_argument.as_deref().is_none_or(str::is_empty)
                {
                    return Err("causes requires temporal precedence and a causal_argument".into());
                }
            }
            EvidenceRelation::HistoricallyInfluences => {
                if !strictly_precedes(from_context, to_context)
                    || transmission_chain.as_deref().is_none_or(str::is_empty)
                {
                    return Err("historical influence requires temporal precedence and a transmission_chain".into());
                }
            }
        }

        let symmetric_relation = matches!(
            relation,
            EvidenceRelation::Corroborates | EvidenceRelation::Contradicts | EvidenceRelation::Resembles
        );
        if let Some(existing) = self.evidence_links.iter().find(|link| {
            let same_direction =
                link.from_fact_id == from_fact_id && link.to_fact_id == to_fact_id;
            let reverse_direction = symmetric_relation
                && link.from_fact_id == to_fact_id
                && link.to_fact_id == from_fact_id;
            (same_direction || reverse_direction)
                && link.relation == relation
                && link.status == EvidenceLinkStatus::Adjudicated
        }) {
            if existing.evidence_ids == evidence_ids {
                return Ok(existing.clone());
            }
            return Err(
                "this fact pair/relation is already adjudicated with a different evidence set; update the claim context before adjudicating again"
                    .into(),
            );
        }
        let link = EvidenceLink {
            from_fact_id,
            to_fact_id,
            relation,
            status: EvidenceLinkStatus::Adjudicated,
            evidence_ids: evidence_ids.clone(),
            rationale: Some(rationale),
            causal_argument,
            transmission_chain,
            detected_at_cycle: current_cycle,
        };
        if relation == EvidenceRelation::Corroborates {
            for fact_id in [from_fact_id, to_fact_id] {
                if let Some(fact) = self.facts.get_mut(&fact_id) {
                    fact.corroboration_count = fact.corroboration_count.saturating_add(1);
                    fact.confidence = (fact.confidence + 0.05).clamp(0.0, 1.0);
                    fact.last_accessed_cycle = current_cycle;
                }
            }
        } else if relation == EvidenceRelation::Contradicts {
            for fact_id in [from_fact_id, to_fact_id] {
                if let Some(fact) = self.facts.get_mut(&fact_id) {
                    fact.contradiction_count = fact.contradiction_count.saturating_add(1);
                }
            }
            self.total_contradictions += 1;
        }
        if let Some(candidate_index) = self.evidence_links.iter().position(|candidate| {
            let same_direction =
                candidate.from_fact_id == from_fact_id && candidate.to_fact_id == to_fact_id;
            let reverse_direction = symmetric_relation
                && candidate.from_fact_id == to_fact_id
                && candidate.to_fact_id == from_fact_id;
            (same_direction || reverse_direction)
                && candidate.relation == relation
                && candidate.status == EvidenceLinkStatus::Candidate
        }) {
            self.evidence_links[candidate_index] = link.clone();
        } else {
            self.evidence_links.push(link.clone());
        }
        Ok(link)
    }

    /// Turn a string/HDC contradiction candidate into an evidence-backed decision.
    pub fn adjudicate_contradiction(
        &mut self,
        alert: &ContradictionAlert,
        evidence_ids: Vec<String>,
        rationale: String,
    ) -> Result<(), String> {
        self.adjudicate_evidence_relation(
            alert.existing_fact_id,
            alert.new_fact_id,
            EvidenceRelation::Contradicts,
            evidence_ids,
            rationale,
            None,
            None,
            alert.detected_at_cycle,
        )?;
        Ok(())
    }

    /// Search for facts similar to a query vector, returning top-k results
    pub fn search(
        &mut self,
        query: &BinaryHV,
        k: usize,
        current_cycle: u64,
    ) -> Vec<FactSearchResult> {
        let mut results: Vec<FactSearchResult> = self
            .facts
            .values()
            .map(|fact| {
                let similarity = fact.encoding.vector.similarity(query);
                FactSearchResult {
                    fact_id: fact.id,
                    similarity,
                    confidence: fact.confidence,
                }
            })
            .filter(|r| r.similarity > 0.0) // Only positive similarity
            .collect();

        // Sort by similarity × confidence (relevance-weighted)
        results.sort_by(|a, b| {
            let score_a = a.similarity * a.confidence;
            let score_b = b.similarity * b.confidence;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        // Mark accessed facts as refreshed
        for result in &results {
            if let Some(fact) = self.facts.get_mut(&result.fact_id) {
                fact.last_accessed_cycle = current_cycle;
            }
        }

        results
    }

    /// Search within a specific domain
    pub fn search_domain(&self, domain: &str, query: &BinaryHV, k: usize) -> Vec<FactSearchResult> {
        let fact_ids = match self.domain_index.get(domain) {
            Some(ids) => ids,
            None => return Vec::new(),
        };

        let mut results: Vec<FactSearchResult> = fact_ids
            .iter()
            .filter_map(|id| self.facts.get(id))
            .map(|fact| {
                let similarity = fact.encoding.vector.similarity(query);
                FactSearchResult {
                    fact_id: fact.id,
                    similarity,
                    confidence: fact.confidence,
                }
            })
            .filter(|r| r.similarity > 0.0)
            .collect();

        results.sort_by(|a, b| {
            let score_a = a.similarity * a.confidence;
            let score_b = b.similarity * b.confidence;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }

    /// Apply confidence decay to all facts.
    ///
    /// Called periodically (e.g., every N cycles) to model forgetting.
    /// Facts accessed recently decay less. Evicts facts below threshold.
    pub fn decay_confidence(&mut self, current_cycle: u64) {
        let mut to_evict = Vec::new();

        for fact in self.facts.values_mut() {
            // Recency bonus: recently accessed facts decay slower
            let age_since_access = current_cycle.saturating_sub(fact.last_accessed_cycle);
            let decay = if age_since_access < 100 {
                // Recently accessed: minimal decay
                self.confidence_decay_rate.powf(0.1)
            } else {
                self.confidence_decay_rate
            };

            fact.confidence *= decay;

            // Corroborated facts decay even slower
            if fact.corroboration_count > 0 {
                fact.confidence = fact.confidence.max(fact.initial_confidence * 0.5);
                // Floor at 50% of initial
            }

            // Clinical domain facts decay slower (established clinical knowledge
            // persists longer than episodic observations).
            // Science: clinical ontologies are stable over DSM revision cycles (~15 years).
            #[cfg(feature = "therapeutic")]
            if fact.domain.as_deref() == Some("clinical") {
                fact.confidence = fact.confidence.max(fact.initial_confidence * 0.8);
                // Floor at 80% of initial — clinical facts resist forgetting
            }

            if fact.confidence < self.eviction_threshold {
                to_evict.push(fact.id);
            }
        }

        for id in &to_evict {
            self.remove_fact(*id);
            self.total_evictions += 1;
        }
    }

    /// Get a fact by ID
    pub fn get_fact(&self, id: FactId) -> Option<&TemporalFact> {
        self.facts.get(&id)
    }

    /// Get all facts with causal relations (for DAG construction)
    pub fn causal_facts(&self) -> Vec<&TemporalFact> {
        self.facts
            .values()
            .filter(|f| f.has_causal_relations)
            .collect()
    }

    /// Drain pending contradiction alerts
    pub fn drain_contradictions(&mut self) -> Vec<ContradictionAlert> {
        std::mem::take(&mut self.pending_contradictions)
    }

    /// Number of facts currently stored
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Whether the graph is empty
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    /// Average confidence across all facts
    pub fn average_confidence(&self) -> f32 {
        if self.facts.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.facts.values().map(|f| f.confidence).sum();
        sum / self.facts.len() as f32
    }

    /// Number of distinct domains
    pub fn domain_count(&self) -> usize {
        self.domain_index.len()
    }

    pub fn total_insertions(&self) -> u64 {
        self.total_insertions
    }

    pub fn total_evictions(&self) -> u64 {
        self.total_evictions
    }

    pub fn total_contradictions(&self) -> u64 {
        self.total_contradictions
    }

    // ── Iteration ───────────────────────────────────────────────────────

    /// Iterate over all stored facts.
    pub fn all_facts(&self) -> impl Iterator<Item = &TemporalFact> {
        self.facts.values()
    }

    /// Get per-domain distribution: (domain_name, avg_confidence, fact_count).
    ///
    /// Sorted by average confidence ascending (most uncertain first).
    pub fn domain_distribution(&self) -> Vec<(String, f32, usize)> {
        let mut distribution: Vec<(String, f32, usize)> = self
            .domain_index
            .iter()
            .map(|(domain, ids)| {
                let valid_facts: Vec<f32> = ids
                    .iter()
                    .filter_map(|id| self.facts.get(id))
                    .map(|f| f.confidence)
                    .collect();
                let count = valid_facts.len();
                let avg_conf = if count > 0 {
                    valid_facts.iter().sum::<f32>() / count as f32
                } else {
                    0.0
                };
                (domain.clone(), avg_conf, count)
            })
            .collect();

        distribution.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distribution
    }

    /// Adjust confidence of a specific fact by a delta.
    ///
    /// Clamps result to [0.0, 1.0]. Used by calibration feedback.
    /// Science: Jaynes (2003) — probability as extended logic.
    pub fn adjust_confidence(&mut self, fact_id: FactId, delta: f32) {
        if let Some(fact) = self.facts.get_mut(&fact_id) {
            fact.confidence = (fact.confidence + delta).clamp(0.0, 1.0);
        }
    }

    /// Spreading activation search: expand from seed results through HDC similarity.
    ///
    /// Starting from `seeds`, finds facts similar to each seed (within `hops` rounds),
    /// decaying activation by `decay_factor` per hop. Returns up to `max_results`.
    /// Science: Anderson (1983) — ACT-R spreading activation.
    pub fn spreading_activation_search(
        &mut self,
        seeds: &[FactSearchResult],
        hops: usize,
        decay_factor: f32,
        max_results: usize,
        current_cycle: u64,
    ) -> Vec<FactSearchResult> {
        let mut activated: HashMap<FactId, f32> = HashMap::new();
        let mut frontier_ids: Vec<FactId> = seeds.iter().map(|s| s.fact_id).collect();

        // Seed activation
        for seed in seeds {
            activated.insert(seed.fact_id, seed.similarity);
        }

        for hop in 0..hops {
            let hop_decay = decay_factor.powi(hop as i32 + 1);
            let mut next_frontier = Vec::new();

            for &fid in &frontier_ids {
                let query_vec = match self.facts.get(&fid) {
                    Some(f) => f.encoding.vector.clone(),
                    None => continue,
                };

                // Find similar facts
                for fact in self.facts.values() {
                    if activated.contains_key(&fact.id) {
                        continue;
                    }
                    let sim = fact.encoding.vector.similarity(&query_vec);
                    if sim > 0.1 {
                        let decayed_sim = sim * hop_decay;
                        activated.insert(fact.id, decayed_sim);
                        next_frontier.push(fact.id);
                    }
                }
            }

            frontier_ids = next_frontier;
            if frontier_ids.is_empty() {
                break;
            }
        }

        // Remove seeds from results (caller already has them)
        let seed_ids: std::collections::HashSet<FactId> = seeds.iter().map(|s| s.fact_id).collect();

        let mut results: Vec<FactSearchResult> = activated
            .into_iter()
            .filter(|(id, _)| !seed_ids.contains(id))
            .map(|(id, activation)| {
                let confidence = self.facts.get(&id).map(|f| f.confidence).unwrap_or(0.0);
                FactSearchResult {
                    fact_id: id,
                    similarity: activation,
                    confidence,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            let score_a = a.similarity * a.confidence;
            let score_b = b.similarity * b.confidence;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(max_results);

        // Mark accessed
        for result in &results {
            if let Some(fact) = self.facts.get_mut(&result.fact_id) {
                fact.last_accessed_cycle = current_cycle;
            }
        }

        results
    }

    // ── Contradiction Resolution ────────────────────────────────────────

    /// Resolve contradictions by demoting the weaker fact's confidence.
    /// Science: AGM theory (Alchourrón et al. 1985) — belief contraction.
    pub fn resolve_contradictions(&mut self, alerts: &[ContradictionAlert]) -> usize {
        let mut resolved = 0;
        for alert in alerts {
            let Some(evidence_ids) = self
                .evidence_links
                .iter()
                .find(|link| {
                    let same_direction = link.from_fact_id == alert.existing_fact_id
                        && link.to_fact_id == alert.new_fact_id;
                    // Contradicts is symmetric: adjudication may be recorded from
                    // either witness toward the other, but resolution must still apply
                    // to the confidence ordering carried by this alert.
                    let reverse_direction = link.from_fact_id == alert.new_fact_id
                        && link.to_fact_id == alert.existing_fact_id;
                    (same_direction || reverse_direction)
                        && link.relation == EvidenceRelation::Contradicts
                        && link.status == EvidenceLinkStatus::Adjudicated
                        && !link.evidence_ids.is_empty()
                        && link.rationale.as_deref().is_some_and(|text| !text.trim().is_empty())
                })
                .map(|link| link.evidence_ids.clone())
            else {
                continue;
            };
            let resolution_key = (
                alert.new_fact_id,
                alert.existing_fact_id,
                evidence_ids,
            );
            if self.resolved_contradictions.contains(&resolution_key) {
                continue;
            }
            let weaker_id = if alert.higher_confidence_id == alert.new_fact_id {
                alert.existing_fact_id
            } else {
                alert.new_fact_id
            };
            let mut resolution_applied = false;
            let should_remove = if let Some(fact) = self.facts.get_mut(&weaker_id) {
                fact.confidence *= 0.5;
                resolved += 1;
                resolution_applied = true;
                fact.confidence < 0.05
            } else {
                false
            };
            if resolution_applied {
                self.resolved_contradictions.insert(resolution_key);
            }
            if should_remove {
                self.remove_fact(weaker_id);
            }
        }
        resolved
    }

    // ── Persistence Support ─────────────────────────────────────────────

    /// Import a legacy fact record, assigning an in-memory identity.
    pub fn import_fact_record(&mut self, record: &super::persistence::FactRecord) {
        self.import_fact_record_with_provenance(record, None, None);
    }

    /// Import a graph snapshot record with stable identity and optional provenance.
    pub fn import_graph_fact_record(&mut self, record: &super::persistence::GraphFactRecord) {
        self.import_fact_record_with_provenance(
            &record.fact,
            record.fact_id,
            record.claim_context.clone(),
        );
    }

    fn import_fact_record_with_provenance(
        &mut self,
        record: &super::persistence::FactRecord,
        fact_id: Option<FactId>,
        claim_context: Option<ClaimContext>,
    ) {
        if record.vector_bytes.len() != 2048 {
            return;
        }
        let mut arr = [0u8; 2048];
        arr.copy_from_slice(&record.vector_bytes);
        let encoding = super::encoding::FactEncoding {
            vector: symthaea_core::hdc::binary_hv::BinaryHV(arr),
            role_vectors: std::collections::HashMap::new(),
            source_text: record.source_text.clone(),
            confidence: record.confidence,
        };
        let id = fact_id
            .filter(|candidate| !self.facts.contains_key(candidate))
            .unwrap_or(self.next_id);
        self.next_id = self.next_id.max(id.saturating_add(1));
        let fact = TemporalFact {
            id,
            encoding,
            inserted_at_cycle: record.cycle,
            last_accessed_cycle: record.cycle,
            confidence: record.confidence,
            initial_confidence: record.confidence,
            corroboration_count: 0,
            contradiction_count: 0,
            domain: record.domain.clone(),
            has_causal_relations: record.is_causal,
        };
        if let Some(context) = claim_context {
            self.claim_contexts.insert(id, context);
        }
        if let Some(ref domain) = fact.domain {
            self.domain_index
                .entry(domain.clone())
                .or_default()
                .push(id);
        }
        self.facts.insert(id, fact);
    }

    /// Export all facts as persistence records.
    pub fn export_fact_records(&self) -> Vec<super::persistence::FactRecord> {
        self.facts
            .values()
            .map(|f| super::persistence::FactRecord {
                vector_bytes: f.encoding.vector.0.to_vec(),
                source_text: f.encoding.source_text.clone(),
                confidence: f.confidence,
                domain: f.domain.clone(),
                cycle: f.inserted_at_cycle,
                is_causal: f.has_causal_relations,
            })
            .collect()
    }

    /// Export stable graph identities alongside the legacy persistence record shape.
    pub fn export_graph_fact_records(&self) -> Vec<super::persistence::GraphFactRecord> {
        self.facts
            .values()
            .map(|fact| super::persistence::GraphFactRecord {
                fact_id: Some(fact.id),
                fact: super::persistence::FactRecord {
                    vector_bytes: fact.encoding.vector.0.to_vec(),
                    source_text: fact.encoding.source_text.clone(),
                    confidence: fact.confidence,
                    domain: fact.domain.clone(),
                    cycle: fact.inserted_at_cycle,
                    is_causal: fact.has_causal_relations,
                },
                claim_context: self.claim_contexts.get(&fact.id).cloned(),
            })
            .collect()
    }

    // ── Dream Consolidation ─────────────────────────────────────────────

    /// Prune non-causal facts below confidence threshold.
    /// Science: Anderson & Schooler (1991) — power law of forgetting.
    pub fn prune_low_confidence(&mut self, threshold: f32) -> usize {
        let ids: Vec<FactId> = self
            .facts
            .iter()
            .filter(|(_, f)| f.confidence < threshold && !f.has_causal_relations)
            .map(|(&id, _)| id)
            .collect();
        let count = ids.len();
        for id in ids {
            self.remove_fact(id);
        }
        count
    }

    /// Strengthen causal facts, capped at initial_confidence.
    /// Science: Stickgold & Walker (2013) — sleep-dependent consolidation.
    pub fn strengthen_causal_facts(&mut self, boost: f32) -> usize {
        let mut count = 0;
        for fact in self.facts.values_mut() {
            if fact.has_causal_relations && fact.confidence < fact.initial_confidence {
                fact.confidence = (fact.confidence + boost).min(fact.initial_confidence);
                count += 1;
            }
        }
        count
    }

    /// Strengthen facts whose source text contains any of the given topic keywords.
    ///
    /// Used by the Dream→Knowledge feedback loop: when dream replay consolidates
    /// memories around certain topics, the corresponding knowledge graph edges are
    /// strengthened proportional to the replay quality.
    ///
    /// - `topics`: keywords extracted from consolidated dream content.
    /// - `boost`: confidence increment per matched fact (capped at `initial_confidence`).
    ///
    /// Returns the number of facts strengthened.
    ///
    /// Science: Rasch & Born (2013) — targeted memory reactivation selectively
    /// strengthens cued memory traces during sleep consolidation.
    pub fn strengthen_facts_by_topic(&mut self, topics: &[String], boost: f32) -> usize {
        if topics.is_empty() || boost <= 0.0 {
            return 0;
        }
        let lower_topics: Vec<String> = topics.iter().map(|t| t.to_lowercase()).collect();
        let mut count = 0;
        for fact in self.facts.values_mut() {
            let src = fact.encoding.source_text.to_lowercase();
            let matches = lower_topics
                .iter()
                .any(|t| !t.is_empty() && src.contains(t.as_str()));
            if matches && fact.confidence < fact.initial_confidence {
                fact.confidence = (fact.confidence + boost).min(fact.initial_confidence);
                count += 1;
            }
        }
        count
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn detect_contradictions(
        &self,
        new_encoding: &FactEncoding,
        current_cycle: u64,
    ) -> Vec<ContradictionAlert> {
        let mut alerts = Vec::new();

        for existing in self.facts.values() {
            let sim = new_encoding.vector.similarity(&existing.encoding.vector);

            // High similarity + one is negated = contradiction candidate
            // We approximate this by checking if similarity is high but
            // source texts suggest opposition (simple heuristic)
            if sim > self.contradiction_threshold {
                // Check for negation markers in either text
                let new_lower = new_encoding.source_text.to_lowercase();
                let existing_lower = existing.encoding.source_text.to_lowercase();

                let new_negated = contains_negation(&new_lower);
                let existing_negated = contains_negation(&existing_lower);

                if new_negated != existing_negated {
                    // One is negated and the other isn't = contradiction
                    let higher_confidence_id = if new_encoding.confidence > existing.confidence {
                        self.next_id
                    } else {
                        existing.id
                    };

                    alerts.push(ContradictionAlert {
                        new_fact_id: self.next_id,
                        existing_fact_id: existing.id,
                        similarity: sim,
                        higher_confidence_id,
                        detected_at_cycle: current_cycle,
                    });
                }
            }
        }

        alerts
    }

    fn evict_lowest_confidence(&mut self) {
        if let Some((&id, _)) = self.facts.iter().min_by(|(_, a), (_, b)| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            self.remove_fact(id);
            self.total_evictions += 1;
        }
    }

    fn remove_fact(&mut self, id: FactId) {
        if let Some(fact) = self.facts.remove(&id) {
            self.claim_contexts.remove(&id);
            self.evidence_links
                .retain(|link| link.from_fact_id != id && link.to_fact_id != id);
            self.resolved_contradictions
                .retain(|(new_id, existing_id, _)| *new_id != id && *existing_id != id);
            // Clean domain index
            if let Some(ref domain) = fact.domain {
                if let Some(ids) = self.domain_index.get_mut(domain) {
                    ids.retain(|&fid| fid != id);
                    if ids.is_empty() {
                        self.domain_index.remove(domain);
                    }
                }
            }
        }
    }
}

fn candidate_relation(
    new_context: Option<&ClaimContext>,
    old_context: Option<&ClaimContext>,
) -> (EvidenceRelation, Vec<String>) {
    let mut evidence_ids = Vec::new();
    if let Some(context) = old_context {
        evidence_ids.extend(context.evidence_ids.iter().cloned());
    }
    if let Some(context) = new_context {
        evidence_ids.extend(context.evidence_ids.iter().cloned());
    }
    evidence_ids.sort();
    evidence_ids.dedup();

    let relation = match (new_context, old_context) {
        (Some(new_context), Some(old_context)) if old_context.same_claim(new_context) => {
            if old_context.admissibly_independent_from(new_context) {
                EvidenceRelation::Corroborates
            } else {
                EvidenceRelation::SameWitnessVariant
            }
        }
        (Some(new_context), Some(old_context))
            if old_context.same_conflict_scope(new_context) =>
        {
            EvidenceRelation::Contradicts
        }
        _ => EvidenceRelation::SemanticNeighbor,
    };
    (relation, evidence_ids)
}

fn strictly_precedes(from: &ClaimContext, to: &ClaimContext) -> bool {
    matches!((from.event_time, to.event_time), (Some(from_time), Some(to_time)) if from_time < to_time)
}

fn contains_negation(text: &str) -> bool {
    let markers = [
        "not ",
        "no ",
        "never ",
        "cannot ",
        "can't ",
        "won't ",
        "didn't ",
        "doesn't ",
        "isn't ",
        "aren't ",
        "without ",
        "failed to ",
        "unable to ",
    ];
    markers.iter().any(|m| text.contains(m))
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::claim_provenance::{ClaimKind, ClaimPolarity};

    fn make_encoding(text: &str, confidence: f32) -> FactEncoding {
        FactEncoding {
            vector: BinaryHV::random(crate::knowledge::encoding::fnv1a_hash(text)),
            role_vectors: HashMap::new(),
            source_text: text.to_string(),
            confidence,
        }
    }

    fn make_claim_context(
        witness_id: &str,
        source_family_id: &str,
        evidence_id: &str,
        object: &str,
        polarity: ClaimPolarity,
        event_time: Option<i64>,
    ) -> ClaimContext {
        ClaimContext {
            subject: "Oil prices".into(),
            predicate: "rose".into(),
            object: object.into(),
            witness_id: witness_id.into(),
            source_family_id: source_family_id.into(),
            context_id: "market-report:day-1".into(),
            temporal_scope: Some("day-1".into()),
            event_time,
            claim_kind: ClaimKind::SourceAttested,
            polarity,
            evidence_ids: vec![evidence_id.into()],
        }
    }

    #[test]
    fn test_insert_and_search() {
        let mut graph = EnhancedKnowledgeGraph::new(100);

        let enc = make_encoding("Iran launched missiles", 0.8);
        let query = enc.vector.clone();
        let (id, _) = graph.insert(enc, 1, Some("geopolitics".to_string()), true);

        assert_eq!(graph.len(), 1);
        let results = graph.search(&query, 5, 2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].fact_id, id);
    }

    #[test]
    fn test_confidence_decay() {
        let mut graph = EnhancedKnowledgeGraph::new(100);

        let enc = make_encoding("test fact", 0.8);
        let (id, _) = graph.insert(enc, 1, None, false);

        // Apply many decay cycles
        for i in 0..10000 {
            graph.decay_confidence(100 + i);
        }

        // Confidence should have decreased
        let fact = graph.get_fact(id);
        if let Some(f) = fact {
            assert!(f.confidence < 0.8);
        }
        // Very old, unaccessed facts may be evicted
    }

    #[test]
    fn semantic_similarity_does_not_merge_or_boost_generic_facts() {
        let mut graph = EnhancedKnowledgeGraph::new(100);

        let (id1, _) = graph.insert(make_encoding("oil prices rose", 0.7), 1, None, false);
        let before = graph.get_fact(id1).unwrap().confidence;
        let (id2, _) = graph.insert(make_encoding("oil prices rose", 0.8), 2, None, false);

        assert_ne!(id1, id2);
        assert_eq!(graph.len(), 2);
        assert_eq!(graph.get_fact(id1).unwrap().corroboration_count, 0);
        assert_eq!(graph.get_fact(id1).unwrap().confidence, before);
        assert!(graph.evidence_links().iter().any(|link| {
            link.from_fact_id == id1
                && link.to_fact_id == id2
                && link.relation == EvidenceRelation::SemanticNeighbor
                && link.status == EvidenceLinkStatus::Candidate
        }));
    }

    #[test]
    fn grounded_similarity_creates_candidate_not_corroboration() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let context_a = make_claim_context(
            "witness-a", "family-a", "evidence-a", "renewal", ClaimPolarity::Positive, Some(1),
        );
        let context_b = make_claim_context(
            "witness-b", "family-b", "evidence-b", "renewal", ClaimPolarity::Positive, Some(1),
        );
        let (id_a, _) = graph
            .insert_grounded(make_encoding("oil prices rose", 0.7), 1, None, false, context_a)
            .unwrap();
        let (id_b, _) = graph
            .insert_grounded(make_encoding("oil prices rose", 0.7), 2, None, false, context_b)
            .unwrap();

        let link = graph.evidence_links().iter().find(|link| {
            link.from_fact_id == id_a && link.to_fact_id == id_b
        }).unwrap();
        assert_eq!(link.relation, EvidenceRelation::Corroborates);
        assert_eq!(link.status, EvidenceLinkStatus::Candidate);
        assert_eq!(graph.get_fact(id_a).unwrap().corroboration_count, 0);
        assert_eq!(graph.get_fact(id_a).unwrap().confidence, 0.7);
        assert_eq!(graph.get_fact(id_b).unwrap().confidence, 0.7);
    }

    #[test]
    fn rejecting_similarity_candidate_does_not_change_confidence() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let (first, _) = graph.insert(make_encoding("symbol means renewal", 0.7), 1, None, false);
        let before = graph.get_fact(first).unwrap().confidence;
        let (second, _) = graph.insert(make_encoding("symbol means renewal", 0.8), 2, None, false);
        let rejected = graph
            .reject_evidence_candidate(
                first,
                second,
                EvidenceRelation::SemanticNeighbor,
                "Same vector pattern but no admissible identity evidence".into(),
                3,
            )
            .unwrap();
        assert_eq!(rejected.status, EvidenceLinkStatus::Rejected);
        assert_eq!(graph.get_fact(first).unwrap().confidence, before);
        assert_eq!(graph.get_fact(first).unwrap().corroboration_count, 0);
    }

    #[test]
    fn explicit_independent_corroboration_boosts_once() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let context_a = make_claim_context(
            "witness-a", "family-a", "evidence-a", "renewal", ClaimPolarity::Positive, Some(1),
        );
        let context_b = make_claim_context(
            "witness-b", "family-b", "evidence-b", "renewal", ClaimPolarity::Positive, Some(1),
        );
        let (id_a, _) = graph
            .insert_grounded(make_encoding("oil prices rose", 0.7), 1, None, false, context_a)
            .unwrap();
        let (id_b, _) = graph
            .insert_grounded(make_encoding("oil prices rose", 0.7), 2, None, false, context_b)
            .unwrap();
        let evidence = vec!["evidence-a".to_string(), "evidence-b".to_string()];

        let link = graph.adjudicate_evidence_relation(
            id_a,
            id_b,
            EvidenceRelation::Corroborates,
            evidence.clone(),
            "Independent witnesses support the same scoped proposition".into(),
            None,
            None,
            3,
        ).unwrap();
        assert_eq!(link.status, EvidenceLinkStatus::Adjudicated);
        assert_eq!(graph.get_fact(id_a).unwrap().corroboration_count, 1);
        assert_eq!(graph.get_fact(id_a).unwrap().confidence, 0.75);

        graph.adjudicate_evidence_relation(
            id_a,
            id_b,
            EvidenceRelation::Corroborates,
            evidence,
            "Same evidence adjudication repeated".into(),
            None,
            None,
            4,
        ).unwrap();
        assert_eq!(graph.get_fact(id_a).unwrap().corroboration_count, 1);
        assert_eq!(graph.get_fact(id_a).unwrap().confidence, 0.75);
    }

    #[test]
    fn imported_evidence_links_reject_unbound_adjudication_ids_atomically() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let context = |witness: &str, family: &str, evidence: &str| ClaimContext {
            subject: "subject".into(),
            predicate: "predicate".into(),
            object: "object".into(),
            witness_id: witness.into(),
            source_family_id: family.into(),
            context_id: "shared-context".into(),
            temporal_scope: Some("period-a".into()),
            event_time: Some(1),
            claim_kind: ClaimKind::SourceAttested,
            polarity: ClaimPolarity::Positive,
            evidence_ids: vec![evidence.into()],
        };
        let (a, _) = graph
            .insert_grounded(
                make_encoding("claim a", 0.5),
                1,
                None,
                false,
                context("witness-a", "family-a", "evidence-a"),
            )
            .unwrap();
        let (b, _) = graph
            .insert_grounded(
                make_encoding("claim b", 0.5),
                2,
                None,
                false,
                context("witness-b", "family-b", "evidence-b"),
            )
            .unwrap();
        let invalid = EvidenceLink {
            from_fact_id: a,
            to_fact_id: b,
            relation: EvidenceRelation::Corroborates,
            status: EvidenceLinkStatus::Adjudicated,
            evidence_ids: vec![
                "evidence-a".into(),
                "evidence-b".into(),
                "unbound-id".into(),
            ],
            rationale: Some("malformed persisted-link control".into()),
            causal_argument: None,
            transmission_chain: None,
            detected_at_cycle: 3,
        };
        let before = graph.evidence_links().len();
        let error = graph.import_evidence_links(&[invalid]).unwrap_err();
        assert!(error.contains("not declared by either claim context"));
        assert_eq!(graph.evidence_links().len(), before);
        assert_eq!(graph.get_fact(a).unwrap().corroboration_count, 0);
        assert_eq!(graph.get_fact(b).unwrap().corroboration_count, 0);
    }

    #[test]
    fn adjudication_rejects_unbound_extra_evidence_ids() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let context_a = make_claim_context(
            "witness-a",
            "family-a",
            "evidence-a",
            "renewal",
            ClaimPolarity::Positive,
            Some(1),
        );
        let context_b = make_claim_context(
            "witness-b",
            "family-b",
            "evidence-b",
            "renewal",
            ClaimPolarity::Positive,
            Some(1),
        );
        let (id_a, _) = graph
            .insert_grounded(
                make_encoding("same proposition", 0.7),
                1,
                None,
                false,
                context_a,
            )
            .unwrap();
        let (id_b, _) = graph
            .insert_grounded(
                make_encoding("same proposition", 0.7),
                2,
                None,
                false,
                context_b,
            )
            .unwrap();

        let error = graph
            .adjudicate_evidence_relation(
                id_a,
                id_b,
                EvidenceRelation::Corroborates,
                vec![
                    "evidence-a".into(),
                    "evidence-b".into(),
                    "unbound-evidence".into(),
                ],
                "Attempt to attach an evidence ID absent from either claim context".into(),
                None,
                None,
                3,
            )
            .unwrap_err();
        assert!(error.contains("not declared by either claim context"));
        assert_eq!(graph.get_fact(id_a).unwrap().corroboration_count, 0);
        assert_eq!(graph.get_fact(id_b).unwrap().corroboration_count, 0);
        assert!(graph.evidence_links().is_empty());
    }

    #[test]
    fn corroboration_rejects_two_witnesses_from_one_source_family() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let context_a = make_claim_context(
            "witness-a",
            "same-work-family",
            "evidence-a",
            "renewal",
            ClaimPolarity::Positive,
            Some(1),
        );
        let context_b = make_claim_context(
            "witness-b",
            "same-work-family",
            "evidence-b",
            "renewal",
            ClaimPolarity::Positive,
            Some(1),
        );
        let (id_a, _) = graph
            .insert_grounded(
                make_encoding("same proposition", 0.7),
                1,
                None,
                false,
                context_a,
            )
            .unwrap();
        let (id_b, _) = graph
            .insert_grounded(
                make_encoding("same proposition", 0.7),
                2,
                None,
                false,
                context_b,
            )
            .unwrap();
        let error = graph
            .adjudicate_evidence_relation(
                id_a,
                id_b,
                EvidenceRelation::Corroborates,
                vec!["evidence-a".into(), "evidence-b".into()],
                "Two witnesses from one shared source family".into(),
                None,
                None,
                3,
            )
            .unwrap_err();
        assert!(error.contains("independent source evidence"));
        assert_eq!(graph.get_fact(id_a).unwrap().corroboration_count, 0);
        assert_eq!(graph.get_fact(id_a).unwrap().confidence, 0.7);
    }

    #[test]
    fn corroboration_cannot_boost_twice_for_reverse_orientation_or_new_evidence_subset() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let mut context_a = make_claim_context(
            "witness-a", "family-a", "evidence-a", "renewal", ClaimPolarity::Positive, Some(1),
        );
        context_a.evidence_ids.push("evidence-a2".into());
        let mut context_b = make_claim_context(
            "witness-b", "family-b", "evidence-b", "renewal", ClaimPolarity::Positive, Some(1),
        );
        context_b.evidence_ids.push("evidence-b2".into());
        let (id_a, _) = graph
            .insert_grounded(make_encoding("same proposition", 0.7), 1, None, false, context_a)
            .unwrap();
        let (id_b, _) = graph
            .insert_grounded(make_encoding("same proposition", 0.7), 2, None, false, context_b)
            .unwrap();

        graph.adjudicate_evidence_relation(
            id_a, id_b, EvidenceRelation::Corroborates,
            vec!["evidence-a".into(), "evidence-b".into()],
            "Two independent source families".into(), None, None, 3,
        ).unwrap();
        assert!(
            graph.adjudicate_evidence_relation(
                id_b, id_a, EvidenceRelation::Corroborates,
                vec!["evidence-a".into(), "evidence-b".into()],
                "Same adjudication requested in reverse".into(), None, None, 4,
            ).is_ok()
        );
        let error = graph.adjudicate_evidence_relation(
            id_a, id_b, EvidenceRelation::Corroborates,
            vec!["evidence-a2".into(), "evidence-b2".into()],
            "Alternative subset from same two sources".into(), None, None, 5,
        ).unwrap_err();
        assert!(error.contains("already adjudicated with a different evidence set"));
        assert_eq!(graph.get_fact(id_a).unwrap().corroboration_count, 1);
        assert_eq!(graph.get_fact(id_a).unwrap().confidence, 0.75);
        assert_eq!(graph.evidence_links().iter().filter(|link| {
            link.relation == EvidenceRelation::Corroborates
                && link.status == EvidenceLinkStatus::Adjudicated
        }).count(), 1);
    }

    #[test]
    fn historical_influence_requires_explicit_transmission_chain() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let earlier = make_claim_context(
            "witness-a", "family-a", "evidence-a", "renewal", ClaimPolarity::Positive, Some(-300),
        );
        let later = make_claim_context(
            "witness-b", "family-b", "evidence-b", "renewal", ClaimPolarity::Positive, Some(-200),
        );
        let (from, _) = graph.insert_grounded(
            make_encoding("earlier myth", 0.7), 1, None, false, earlier,
        ).unwrap();
        let (to, _) = graph.insert_grounded(
            make_encoding("later myth", 0.7), 2, None, false, later,
        ).unwrap();
        let error = graph.adjudicate_evidence_relation(
            from, to, EvidenceRelation::HistoricallyInfluences,
            vec!["evidence-a".into(), "evidence-b".into()],
            "Motif similarity only".into(), None, None, 3,
        ).unwrap_err();
        assert!(error.contains("transmission_chain"));

        assert!(graph.adjudicate_evidence_relation(
            from, to, EvidenceRelation::HistoricallyInfluences,
            vec!["evidence-a".into(), "evidence-b".into()],
            "Documented transmission argument".into(), None,
            Some("source A -> intermediary -> source B".into()), 3,
        ).is_ok());
    }

    #[test]
    fn graph_snapshot_round_trip_preserves_claim_context_and_candidate_links() {
        let mut original = EnhancedKnowledgeGraph::new(100);
        let context_a = make_claim_context(
            "witness-a",
            "family-a",
            "evidence-a",
            "renewal",
            ClaimPolarity::Positive,
            Some(-300),
        );
        let context_b = make_claim_context(
            "witness-b",
            "family-b",
            "evidence-b",
            "renewal",
            ClaimPolarity::Positive,
            Some(-300),
        );
        let (id_a, _) = original
            .insert_grounded(
                make_encoding("same grounded claim", 0.7),
                1,
                None,
                false,
                context_a,
            )
            .unwrap();
        let (id_b, _) = original
            .insert_grounded(
                make_encoding("same grounded claim", 0.7),
                2,
                None,
                false,
                context_b,
            )
            .unwrap();
        let records = original.export_graph_fact_records();
        let links = original.evidence_links().to_vec();
        assert_eq!(records.len(), 2);
        assert_eq!(links.len(), 1);

        let mut restored = EnhancedKnowledgeGraph::new(100);
        for record in &records {
            restored.import_graph_fact_record(record);
        }
        restored.import_evidence_links(&links).unwrap();
        assert_eq!(restored.get_fact(id_a).unwrap().confidence, 0.7);
        assert_eq!(restored.get_fact(id_b).unwrap().confidence, 0.7);
        let adjudicated = restored
            .adjudicate_evidence_relation(
                id_a,
                id_b,
                EvidenceRelation::Corroborates,
                vec!["evidence-a".into(), "evidence-b".into()],
                "Two independent witnesses support the same scoped proposition".into(),
                None,
                None,
                3,
            )
            .unwrap();
        assert_eq!(adjudicated.status, EvidenceLinkStatus::Adjudicated);
        assert_eq!(restored.get_fact(id_a).unwrap().corroboration_count, 1);
        assert_eq!(restored.get_fact(id_a).unwrap().confidence, 0.75);
    }

    #[test]
    fn contradiction_candidates_do_not_demote_facts_before_adjudication() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let (id_a, _) = graph.insert(make_encoding("oil prices rose", 0.8), 1, None, false);
        let before = graph.get_fact(id_a).unwrap().confidence;
        let mut negative = make_encoding("oil prices did not rise", 0.8);
        negative.vector = graph.get_fact(id_a).unwrap().encoding.vector.clone();
        let (id_b, alerts) = graph.insert(negative, 2, None, false);
        assert_eq!(alerts.len(), 1);
        assert_eq!(graph.resolve_contradictions(&alerts), 0);
        assert_eq!(graph.get_fact(id_a).unwrap().confidence, before);
        assert_eq!(graph.get_fact(id_a).unwrap().contradiction_count, 0);
        assert!(graph.get_fact(id_b).is_some());
    }

    #[test]
    fn contradiction_resolution_requires_adjudication_and_runs_once() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let positive_context = make_claim_context(
            "witness-a",
            "family-a",
            "evidence-a",
            "rise",
            ClaimPolarity::Positive,
            Some(1),
        );
        let negative_context = make_claim_context(
            "witness-b",
            "family-b",
            "evidence-b",
            "rise",
            ClaimPolarity::Negative,
            Some(1),
        );
        let (positive_id, _) = graph
            .insert_grounded(
                make_encoding("oil prices rose", 0.8),
                1,
                None,
                false,
                positive_context,
            )
            .unwrap();
        let mut negative_encoding = make_encoding("oil prices did not rise", 0.6);
        negative_encoding.vector = graph
            .get_fact(positive_id)
            .unwrap()
            .encoding
            .vector
            .clone();
        let (negative_id, candidates) = graph
            .insert_grounded(negative_encoding, 2, None, false, negative_context)
            .unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(graph.resolve_contradictions(&candidates), 0);
        assert_eq!(graph.get_fact(negative_id).unwrap().confidence, 0.6);

        // Contradiction is symmetric, so deliberately adjudicate from the
        // newly inserted negative claim back toward the existing positive claim.
        graph
            .adjudicate_evidence_relation(
                negative_id,
                positive_id,
                EvidenceRelation::Contradicts,
                vec!["evidence-a".into(), "evidence-b".into()],
                "Both source claims address the same scoped proposition with opposite polarity".into(),
                None,
                None,
                3,
            )
            .unwrap();
        assert_eq!(graph.resolve_contradictions(&candidates), 1);
        assert!((graph.get_fact(negative_id).unwrap().confidence - 0.3).abs() < f32::EPSILON);
        assert_eq!(graph.resolve_contradictions(&candidates), 0);
        assert!((graph.get_fact(negative_id).unwrap().confidence - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_domain_search() {
        let mut graph = EnhancedKnowledgeGraph::new(100);

        let enc1 = make_encoding("economic sanctions", 0.8);
        graph.insert(enc1, 1, Some("economics".to_string()), false);

        let enc2 = make_encoding("military operations", 0.8);
        graph.insert(enc2, 2, Some("military".to_string()), false);

        let query = BinaryHV::random(crate::knowledge::encoding::fnv1a_hash("economic sanctions"));
        let results = graph.search_domain("economics", &query, 5);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_eviction_at_capacity() {
        let mut graph = EnhancedKnowledgeGraph::new(3);

        for i in 0..5 {
            let enc = make_encoding(&format!("fact {i}"), 0.5 + i as f32 * 0.1);
            graph.insert(enc, i as u64, None, false);
        }

        assert!(graph.len() <= 3);
        assert!(graph.total_evictions() > 0);
    }

    #[test]
    fn test_causal_facts_filter() {
        let mut graph = EnhancedKnowledgeGraph::new(100);

        let enc1 = make_encoding("sanctions cause inflation", 0.8);
        graph.insert(enc1, 1, None, true);

        let enc2 = make_encoding("the sky is blue", 0.9);
        graph.insert(enc2, 2, None, false);

        let causal = graph.causal_facts();
        assert_eq!(causal.len(), 1);
    }

    #[test]
    fn test_empty_graph() {
        let graph = EnhancedKnowledgeGraph::new(100);
        assert!(graph.is_empty());
        assert_eq!(graph.average_confidence(), 0.0);
        assert_eq!(graph.domain_count(), 0);
    }
}
