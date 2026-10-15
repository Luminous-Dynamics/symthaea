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

use super::encoding::{FactEncoding, RoleBoundVector};
use std::collections::HashMap;
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

/// Result from matching typed role queries against all stored role observations.
#[derive(Debug, Clone, PartialEq)]
pub struct RoleAwareSearchResult {
    /// Stable graph fact identifier.
    pub fact_id: FactId,
    /// Mean matched cosine similarity, with unmatched query terms contributing zero.
    /// This is a retrieval score, not calibrated epistemic confidence.
    pub similarity: f32,
    /// Current stored fact confidence, returned separately from retrieval similarity.
    pub confidence: f32,
    /// Number of query terms assigned a distinct stored observation.
    pub matched_query_terms: usize,
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
    /// Pending contradiction alerts (drained by the knowledge manager each cycle)
    pending_contradictions: Vec<ContradictionAlert>,
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
            total_insertions: 0,
            total_evictions: 0,
            total_contradictions: 0,
        }
    }

    /// Insert a new fact into the knowledge graph.
    ///
    /// Returns the fact ID and any contradiction alerts generated.
    pub fn insert(
        &mut self,
        encoding: FactEncoding,
        current_cycle: u64,
        domain: Option<String>,
        has_causal: bool,
    ) -> (FactId, Vec<ContradictionAlert>) {
        // Check for contradictions before inserting
        let contradictions = self.detect_contradictions(&encoding, current_cycle);

        // Check for corroboration (highly similar existing fact)
        if let Some(existing_id) = self.find_corroboration(&encoding) {
            if let Some(fact) = self.facts.get_mut(&existing_id) {
                fact.corroboration_count += 1;
                // Boost confidence on corroboration (capped at initial)
                fact.confidence = (fact.confidence + 0.1).min(1.0);
                fact.last_accessed_cycle = current_cycle;
                return (existing_id, contradictions);
            }
        }

        // Evict if at capacity
        if self.facts.len() >= self.capacity {
            self.evict_lowest_confidence();
        }

        let id = self.next_id;
        self.next_id += 1;

        let confidence = encoding.confidence;
        let fact = TemporalFact {
            id,
            encoding,
            inserted_at_cycle: current_cycle,
            last_accessed_cycle: current_cycle,
            confidence,
            initial_confidence: confidence,
            corroboration_count: 0,
            contradiction_count: contradictions.len() as u32,
            domain: domain.clone(),
            has_causal_relations: has_causal,
        };

        self.facts.insert(id, fact);
        self.total_insertions += 1;

        // Update domain index
        if let Some(ref d) = domain {
            self.domain_index.entry(d.clone()).or_default().push(id);
        }

        // Store contradiction alerts
        self.pending_contradictions.extend(contradictions.clone());
        self.total_contradictions += contradictions.len() as u64;

        (id, contradictions)
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

    /// Search using typed role-bound query observations.
    ///
    /// Each query term can match only one stored observation and each stored
    /// observation can satisfy only one query term. This preserves same-role
    /// multiplicity: one stored Agent cannot satisfy two distinct Agent terms.
    ///
    /// Only same-role pairs with bipolar cosine similarity >= 0.1 are matched.
    /// Similarity is a retrieval signal; confidence remains a separate field
    /// and is not multiplied into the ranking score.
    pub fn search_role_observations(
        &mut self,
        query: &[RoleBoundVector],
        k: usize,
        current_cycle: u64,
    ) -> Vec<RoleAwareSearchResult> {
        if query.is_empty() || k == 0 {
            return Vec::new();
        }

        const MIN_ROLE_MATCH_COSINE: f32 = 0.1;
        let mut results = Vec::new();

        for fact in self.facts.values() {
            let observations = &fact.encoding.role_vector_observations;
            if observations.is_empty() {
                continue;
            }

            let mut candidate_pairs = Vec::new();
            for (query_index, query_term) in query.iter().enumerate() {
                for (observation_index, observation) in observations.iter().enumerate() {
                    if query_term.role != observation.role {
                        continue;
                    }
                    let similarity = query_term.vector.cosine_similarity(&observation.vector);
                    if similarity >= MIN_ROLE_MATCH_COSINE {
                        candidate_pairs.push((similarity, query_index, observation_index));
                    }
                }
            }

            candidate_pairs.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
                    .then_with(|| a.2.cmp(&b.2))
            });

            let mut query_used = vec![false; query.len()];
            let mut observation_used = vec![false; observations.len()];
            let mut matched_query_terms = 0usize;
            let mut total_similarity = 0.0f32;

            for (similarity, query_index, observation_index) in candidate_pairs {
                if query_used[query_index] || observation_used[observation_index] {
                    continue;
                }
                query_used[query_index] = true;
                observation_used[observation_index] = true;
                matched_query_terms += 1;
                total_similarity += similarity;
            }

            if matched_query_terms > 0 {
                results.push(RoleAwareSearchResult {
                    fact_id: fact.id,
                    similarity: total_similarity / query.len() as f32,
                    confidence: fact.confidence,
                    matched_query_terms,
                });
            }
        }

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.matched_query_terms.cmp(&a.matched_query_terms))
                .then_with(|| a.fact_id.cmp(&b.fact_id))
        });
        results.truncate(k);

        for result in &results {
            if let Some(fact) = self.facts.get_mut(&result.fact_id) {
                fact.last_accessed_cycle = current_cycle;
            }
        }

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

    /// Return the next graph fact ID that will be assigned.
    pub fn next_fact_id(&self) -> FactId {
        self.next_id
    }

    /// Restore a persisted ID allocator high-water mark without moving it backward.
    ///
    /// IDs that were removed before a snapshot remain retired, preventing stale
    /// external references from accidentally identifying a later, unrelated fact.
    pub fn ensure_next_fact_id(&mut self, reserved_next_id: FactId) {
        self.next_id = self.next_id.max(reserved_next_id.max(1));
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
            let weaker_id = if alert.higher_confidence_id == alert.new_fact_id {
                alert.existing_fact_id
            } else {
                alert.new_fact_id
            };
            let should_remove = if let Some(fact) = self.facts.get_mut(&weaker_id) {
                fact.confidence *= 0.5;
                fact.contradiction_count += 1;
                resolved += 1;
                fact.confidence < 0.05
            } else {
                false
            };
            if should_remove {
                self.remove_fact(weaker_id);
            }
        }
        resolved
    }

    // ── Persistence Support ─────────────────────────────────────────────

    /// Import a fact from a persistence record.
    ///
    /// Preserves its graph-level ID when present. Legacy records without an ID
    /// receive a fresh ID that does not collide with facts already imported.
    pub fn import_fact_record(&mut self, record: &super::persistence::FactRecord) {
        if record.vector_bytes.len() != symthaea_core::hdc::unified_hv::BinaryHV::BYTES
            || record
                .role_vectors
                .values()
                .any(|bytes| bytes.len() != symthaea_core::hdc::unified_hv::BinaryHV::BYTES)
            || record.role_vector_observations.iter().any(|observation| {
                observation.vector_bytes.len()
                    != symthaea_core::hdc::unified_hv::BinaryHV::BYTES
            })
        {
            return;
        }
        let mut arr = [0u8; 2048];
        arr.copy_from_slice(&record.vector_bytes);
        let role_vectors = record
            .role_vectors
            .iter()
            .map(|(role, bytes)| {
                let mut role_arr = [0u8; symthaea_core::hdc::unified_hv::BinaryHV::BYTES];
                role_arr.copy_from_slice(bytes);
                (*role, symthaea_core::hdc::binary_hv::BinaryHV(role_arr))
            })
            .collect();
        let role_vector_observations = record
            .role_vector_observations
            .iter()
            .map(|observation| {
                let mut observation_arr =
                    [0u8; symthaea_core::hdc::unified_hv::BinaryHV::BYTES];
                observation_arr.copy_from_slice(&observation.vector_bytes);
                super::encoding::RoleBoundVector {
                    role: observation.role,
                    entity_text: observation.entity_text.clone(),
                    vector: symthaea_core::hdc::binary_hv::BinaryHV(observation_arr),
                }
            })
            .collect();
        let encoding = super::encoding::FactEncoding {
            vector: symthaea_core::hdc::binary_hv::BinaryHV(arr),
            role_vectors,
            role_vector_observations,
            relations: record.relations.clone(),
            source_text: record.source_text.clone(),
            confidence: record.confidence,
        };

        let id = match record.graph_fact_id {
            Some(id) if id > 0 && !self.facts.contains_key(&id) => id,
            _ => {
                while self.facts.contains_key(&self.next_id) {
                    self.next_id = self.next_id.saturating_add(1);
                }
                let id = self.next_id;
                self.next_id = self.next_id.saturating_add(1);
                id
            }
        };
        self.next_id = self.next_id.max(id.saturating_add(1));

        let fact = TemporalFact {
            id,
            encoding,
            inserted_at_cycle: record.cycle,
            last_accessed_cycle: record.last_accessed_cycle.unwrap_or(record.cycle),
            confidence: record.confidence,
            initial_confidence: record.initial_confidence.unwrap_or(record.confidence),
            corroboration_count: record.corroboration_count,
            contradiction_count: record.contradiction_count,
            domain: record.domain.clone(),
            has_causal_relations: record.is_causal,
        };
        if let Some(ref domain) = fact.domain {
            self.domain_index
                .entry(domain.clone())
                .or_default()
                .push(id);
        }
        self.facts.insert(id, fact);
    }

    /// Import a batch without allowing legacy rows to occupy IDs that belong to
    /// newer records carrying stable graph IDs.
    ///
    /// Records with explicit IDs are imported first; rows from pre-ID schemas
    /// then receive IDs above the highest restored stable ID.
    pub fn import_fact_records(&mut self, records: &[super::persistence::FactRecord]) -> usize {
        let before = self.facts.len();
        for record in records.iter().filter(|record| record.graph_fact_id.is_some()) {
            self.import_fact_record(record);
        }
        for record in records.iter().filter(|record| record.graph_fact_id.is_none()) {
            self.import_fact_record(record);
        }
        self.facts.len().saturating_sub(before)
    }

    /// Export all facts as persistence records.
    pub fn export_fact_records(&self) -> Vec<super::persistence::FactRecord> {
        self.facts
            .values()
            .map(|f| super::persistence::FactRecord {
                vector_bytes: f.encoding.vector.0.to_vec(),
                relations: f.encoding.relations.clone(),
                role_vectors: f
                    .encoding
                    .role_vectors
                    .iter()
                    .map(|(role, vector)| (*role, vector.0.to_vec()))
                    .collect(),
                role_vector_observations: f
                    .encoding
                    .role_vector_observations
                    .iter()
                    .map(|observation| super::persistence::RoleVectorObservationRecord {
                        role: observation.role,
                        entity_text: observation.entity_text.clone(),
                        vector_bytes: observation.vector.0.to_vec(),
                    })
                    .collect(),
                source_text: f.encoding.source_text.clone(),
                confidence: f.confidence,
                domain: f.domain.clone(),
                cycle: f.inserted_at_cycle,
                is_causal: f.has_causal_relations,
                graph_fact_id: Some(f.id),
                initial_confidence: Some(f.initial_confidence),
                last_accessed_cycle: Some(f.last_accessed_cycle),
                corroboration_count: f.corroboration_count,
                contradiction_count: f.contradiction_count,
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
                        0 // Placeholder — will be set after insertion
                    } else {
                        existing.id
                    };

                    alerts.push(ContradictionAlert {
                        new_fact_id: self.next_id, // Will be assigned
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

    fn find_corroboration(&self, encoding: &FactEncoding) -> Option<FactId> {
        // A fact with >0.85 similarity is likely the same information
        for existing in self.facts.values() {
            let sim = encoding.vector.similarity(&existing.encoding.vector);
            if sim > 0.85 {
                return Some(existing.id);
            }
        }
        None
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
    use super::super::encoding::KnowledgeEncoder;
    use super::super::extraction::{EntityType, ExtractedRelation, SemanticRole};
    use super::super::persistence::KnowledgePersistence;

    fn make_encoding(text: &str, confidence: f32) -> FactEncoding {
        FactEncoding {
            vector: BinaryHV::random(crate::knowledge::encoding::fnv1a_hash(text)),
            role_vectors: HashMap::new(),
            role_vector_observations: Vec::new(),
            relations: Vec::new(),
            source_text: text.to_string(),
            confidence,
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
    fn test_role_observation_search_uses_multiplicity_and_separate_confidence() {
        let mut graph = EnhancedKnowledgeGraph::new(10);
        let mut encoder = KnowledgeEncoder::with_seed(42);
        let query = encoder.encode_role_terms(&[
            (
                SemanticRole::Agent,
                "United States",
                EntityType::Organization,
            ),
            (
                SemanticRole::Agent,
                "European Union",
                EntityType::Organization,
            ),
        ]);

        let mut complete = make_encoding("complete two-agent fact", 0.35);
        complete.role_vector_observations = query.clone();
        let (complete_id, _) = graph.insert(complete, 1, None, false);

        let mut partial = make_encoding("partial one-agent fact", 0.99);
        partial.role_vector_observations = vec![query[0].clone()];
        let (partial_id, _) = graph.insert(partial, 2, None, false);

        let results = graph.search_role_observations(&query, 10, 3);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].fact_id, complete_id);
        assert_eq!(results[0].matched_query_terms, 2);
        assert!((results[0].similarity - 1.0).abs() < f32::EPSILON);
        assert!((results[0].confidence - 0.35).abs() < f32::EPSILON);

        let partial_result = results
            .iter()
            .find(|result| result.fact_id == partial_id)
            .expect("one matching Agent should retain partial relevance");
        assert_eq!(partial_result.matched_query_terms, 1);
        assert!((partial_result.similarity - 0.5).abs() < f32::EPSILON);
        assert!((partial_result.confidence - 0.99).abs() < f32::EPSILON);
        assert!(results[0].similarity > partial_result.similarity);

        let mut wrong_role = query[0].clone();
        wrong_role.role = SemanticRole::Patient;
        assert!(
            graph
                .search_role_observations(&[wrong_role], 10, 4)
                .is_empty(),
            "a vector tagged with the wrong semantic role must not match"
        );
    }

    #[test]
    fn test_empty_role_observation_query_returns_no_results() {
        let mut graph = EnhancedKnowledgeGraph::new(10);
        graph.insert(make_encoding("any fact", 0.8), 1, None, false);
        assert!(graph.search_role_observations(&[], 5, 2).is_empty());
        assert!(graph
            .search_role_observations(
                &[RoleBoundVector {
                    role: SemanticRole::Agent,
                    entity_text: "missing".into(),
                    vector: BinaryHV::random(404),
                }],
                0,
                2
            )
            .is_empty());
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
    fn test_corroboration() {
        let mut graph = EnhancedKnowledgeGraph::new(100);

        let enc1 = make_encoding("oil prices rose", 0.7);
        let (id1, _) = graph.insert(enc1, 1, None, false);

        // Insert same fact again — should corroborate, not duplicate
        let enc2 = make_encoding("oil prices rose", 0.8);
        let (id2, _) = graph.insert(enc2, 2, None, false);

        assert_eq!(id1, id2); // Same ID = corroborated
        assert_eq!(graph.len(), 1); // Still one fact
        let fact = graph.get_fact(id1).unwrap();
        assert_eq!(fact.corroboration_count, 1);
    }

    #[test]
    fn test_persistence_records_preserve_role_vectors() {
        use crate::knowledge::extraction::SemanticRole;

        let mut graph = EnhancedKnowledgeGraph::new(10);
        let mut encoding = make_encoding("role-vector persistence fixture", 0.9);
        encoding
            .role_vectors
            .insert(SemanticRole::Agent, BinaryHV::random(71));
        encoding
            .role_vectors
            .insert(SemanticRole::Patient, BinaryHV::random(72));
        let (id, _) = graph.insert(encoding, 4, Some("test".into()), false);

        let exported = graph.export_fact_records();
        assert_eq!(exported.len(), 1);
        let expected = graph.get_fact(id).unwrap().encoding.role_vectors.clone();
        for (role, vector) in &expected {
            assert_eq!(
                exported[0].role_vectors.get(role).unwrap(),
                &vector.0.to_vec()
            );
        }

        let mut restored = EnhancedKnowledgeGraph::new(10);
        restored.import_fact_record(&exported[0]);
        assert_eq!(
            restored.get_fact(id).unwrap().encoding.role_vectors,
            expected
        );
    }

    #[test]
    fn test_persistence_records_preserve_temporal_fact_state() {
        let mut graph = EnhancedKnowledgeGraph::new(10);
        let (id, _) = graph.insert(
            make_encoding("temporal fixture", 0.9),
            7,
            Some("test".to_string()),
            true,
        );
        {
            let fact = graph.facts.get_mut(&id).unwrap();
            fact.confidence = 0.61;
            fact.initial_confidence = 0.94;
            fact.last_accessed_cycle = 29;
            fact.corroboration_count = 5;
            fact.contradiction_count = 2;
        }

        let records = graph.export_fact_records();
        assert_eq!(records.len(), 1);
        let mut restored = EnhancedKnowledgeGraph::new(10);
        restored.import_fact_record(&records[0]);
        let fact = restored.all_facts().next().unwrap();

        assert_eq!(fact.id, id, "graph-level fact IDs must survive export/import");
        assert!((fact.confidence - 0.61).abs() < f32::EPSILON);
        assert!((fact.initial_confidence - 0.94).abs() < f32::EPSILON);
        assert_eq!(fact.inserted_at_cycle, 7);
        assert_eq!(fact.last_accessed_cycle, 29);
        assert_eq!(fact.corroboration_count, 5);
        assert_eq!(fact.contradiction_count, 2);
        assert_eq!(fact.domain.as_deref(), Some("test"));
        assert!(fact.has_causal_relations);
    }

    #[test]
    fn test_batch_import_reserves_stable_ids_before_legacy_rows() {
        use super::super::persistence::FactRecord;

        let stable = FactRecord {
            vector_bytes: vec![1u8; 2048],
            source_text: "stable identity".into(),
            confidence: 0.8,
            domain: None,
            cycle: 2,
            is_causal: false,
            relations: Vec::new(),
            role_vector_observations: Vec::new(),

            graph_fact_id: Some(1),
            initial_confidence: Some(0.8),
            last_accessed_cycle: Some(2),
            corroboration_count: 0,
            contradiction_count: 0,
        };
        let legacy = FactRecord {
            vector_bytes: vec![2u8; 2048],
            source_text: "legacy identity".into(),
            confidence: 0.7,
            domain: None,
            cycle: 9,
            is_causal: false,
            relations: Vec::new(),
            role_vector_observations: Vec::new(),

            graph_fact_id: None,
            initial_confidence: None,
            last_accessed_cycle: None,
            corroboration_count: 0,
            contradiction_count: 0,
        };

        // The legacy row comes first in the source vector to prove input ordering
        // cannot steal a stable ID from a record written by the newer schema.
        let mut graph = EnhancedKnowledgeGraph::new(10);
        assert_eq!(graph.import_fact_records(&[legacy, stable]), 2);
        assert_eq!(
            graph.get_fact(1).unwrap().encoding.source_text,
            "stable identity"
        );
        let legacy_fact = graph
            .all_facts()
            .find(|fact| fact.encoding.source_text == "legacy identity")
            .unwrap();
        assert_eq!(legacy_fact.id, 2);

        let (next_id, _) = graph.insert(
            make_encoding("new insertion after migration", 0.6),
            10,
            None,
            false,
        );
        assert_eq!(next_id, 3, "next ID must advance beyond restored facts");
    }

    #[test]
    fn test_deleted_fact_id_is_not_reused_after_sqlite_restart() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_fact_id_high_water_round_trip_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");

        let mut graph = EnhancedKnowledgeGraph::new(10);
        let (retained_id, _) = graph.insert(
            make_encoding("retained stable fact", 0.9),
            1,
            None,
            false,
        );
        let (retired_id, _) = graph.insert(
            make_encoding("retired high ID fact", 0.02),
            2,
            None,
            false,
        );
        assert_eq!((retained_id, retired_id), (1, 2));
        assert_eq!(graph.prune_low_confidence(0.05), 1);
        assert!(graph.get_fact(retired_id).is_none());
        assert_eq!(graph.next_fact_id(), 3);

        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence
            .replace_knowledge_snapshot(
                &graph.export_fact_records(),
                &[],
                &[],
                graph.next_fact_id(),
            )
            .unwrap();

        let records = persistence.load_facts().unwrap();
        let reserved_next_id = persistence
            .load_next_graph_fact_id()
            .unwrap()
            .expect("a complete snapshot must persist the ID allocator");

        let mut restored = EnhancedKnowledgeGraph::new(10);
        assert_eq!(restored.import_fact_records(&records), 1);
        restored.ensure_next_fact_id(reserved_next_id);
        let (new_id, _) = restored.insert(
            make_encoding("brand new fact after restart", 0.7),
            3,
            None,
            false,
        );
        assert_eq!(
            new_id, 3,
            "the removed fact's ID must remain retired after restart"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sqlite_round_trip_preserves_temporal_fact_state() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_temporal_state_round_trip_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");

        let mut graph = EnhancedKnowledgeGraph::new(10);
        let (id, _) = graph.insert(
            make_encoding("persistent temporal fixture", 0.9),
            7,
            Some("test".to_string()),
            true,
        );
        graph.facts.get_mut(&id).unwrap().encoding.relations = vec![ExtractedRelation {
            subject: "ceasefire".into(),
            predicate: "prevented".into(),
            object: "escalation".into(),
            subject_role: SemanticRole::Agent,
            object_role: SemanticRole::Patient,
            is_causal: true,
            is_negated: true,
            confidence: 0.73,
        }];
        {
            let fact = graph.facts.get_mut(&id).unwrap();
            fact.confidence = 0.61;
            fact.initial_confidence = 0.94;
            fact.last_accessed_cycle = 29;
            fact.corroboration_count = 5;
            fact.contradiction_count = 2;
        }
        let expected_vector = graph.get_fact(id).unwrap().encoding.vector.0.to_vec();

        let mut persistence = KnowledgePersistence::new(&db_path);
        assert_eq!(
            persistence.save_facts(&graph.export_fact_records()).unwrap(),
            1
        );
        let records = persistence.load_facts().unwrap();
        assert_eq!(records.len(), 1);

        let mut restored = EnhancedKnowledgeGraph::new(10);
        restored.import_fact_record(&records[0]);
        let fact = restored
            .all_facts()
            .next()
            .expect("restored graph should have a fact");

        assert_eq!(fact.id, id, "SQLite round trip must preserve graph-level fact IDs");
        assert_eq!(fact.encoding.source_text, "persistent temporal fixture");
        assert_eq!(fact.encoding.vector.0.to_vec(), expected_vector);
        assert_eq!(fact.encoding.relations.len(), 1);
        let relation = &fact.encoding.relations[0];
        assert_eq!(relation.subject, "ceasefire");
        assert_eq!(relation.predicate, "prevented");
        assert_eq!(relation.object, "escalation");
        assert_eq!(relation.subject_role, SemanticRole::Agent);
        assert_eq!(relation.object_role, SemanticRole::Patient);
        assert!(relation.is_causal);
        assert!(relation.is_negated);
        assert!((relation.confidence - 0.73).abs() < f32::EPSILON);
        assert!((fact.confidence - 0.61).abs() < f32::EPSILON);
        assert!((fact.initial_confidence - 0.94).abs() < f32::EPSILON);
        assert_eq!(fact.inserted_at_cycle, 7);
        assert_eq!(fact.last_accessed_cycle, 29);
        assert_eq!(fact.corroboration_count, 5);
        assert_eq!(fact.contradiction_count, 2);
        assert_eq!(fact.domain.as_deref(), Some("test"));
        assert!(fact.has_causal_relations);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_legacy_persistence_record_uses_safe_temporal_defaults() {
        let record = super::super::persistence::FactRecord {
            vector_bytes: vec![0u8; 2048],
            source_text: "legacy fixture".into(),
            confidence: 0.6,
            domain: None,
            cycle: 11,
            is_causal: false,
            relations: Vec::new(),
            role_vector_observations: Vec::new(),

            graph_fact_id: None,
            initial_confidence: None,
            last_accessed_cycle: None,
            corroboration_count: 0,
            contradiction_count: 0,
        };
        let mut graph = EnhancedKnowledgeGraph::new(10);
        graph.import_fact_record(&record);
        let fact = graph.all_facts().next().unwrap();

        assert!((fact.initial_confidence - 0.6).abs() < f32::EPSILON);
        assert_eq!(fact.last_accessed_cycle, 11);
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
