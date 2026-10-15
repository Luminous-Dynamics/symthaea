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

/// One auditable query-to-observation assignment returned by role-aware retrieval.
#[derive(Debug, Clone, PartialEq)]
pub struct RoleObservationMatch {
    /// Semantic role used to constrain the match.
    pub role: SemanticRole,
    /// Entity text in the caller's query.
    pub query_entity_text: String,
    /// Entity text stored on the matched fact observation.
    pub observed_entity_text: String,
    /// Bipolar cosine similarity for this one-to-one assignment.
    pub similarity: f32,
}

/// Validated policy for role-observation matching.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoleObservationSearchConfig {
    min_cosine_similarity: f32,
}

impl RoleObservationSearchConfig {
    /// Construct a search policy with a bipolar-cosine threshold in [-1, 1].
    pub fn new(min_cosine_similarity: f32) -> Result<Self, String> {
        if !min_cosine_similarity.is_finite()
            || !(-1.0..=1.0).contains(&min_cosine_similarity)
        {
            return Err(format!(
                "Role-observation cosine threshold must be finite and in [-1, 1], got {min_cosine_similarity}"
            ));
        }
        Ok(Self { min_cosine_similarity })
    }

    /// Return the minimum bipolar-cosine similarity required for a candidate match.
    pub fn min_cosine_similarity(&self) -> f32 {
        self.min_cosine_similarity
    }
}

impl Default for RoleObservationSearchConfig {
    fn default() -> Self {
        Self {
            min_cosine_similarity: 0.1,
        }
    }
}

/// Result from matching typed role queries against all stored role observations.
#[derive(Debug, Clone, PartialEq)]
pub struct RoleAwareSearchResult {
    /// Stable graph fact identifier.
    pub fact_id: FactId,
    /// Coverage-penalized mean matched cosine similarity.
    ///
    /// This equals coverage multiplied by mean_match_similarity. It is a retrieval
    /// score, not calibrated epistemic confidence.
    pub similarity: f32,
    /// Current stored fact confidence, returned separately from retrieval similarity.
    pub confidence: f32,
    /// Number of query terms assigned a distinct stored observation.
    pub matched_query_terms: usize,
    /// Number of distinct query terms submitted.
    pub query_term_count: usize,
    /// Fraction of query terms assigned a distinct stored observation, in [0, 1].
    pub coverage: f32,
    /// Mean cosine among matched pairs only; this is absent when no result is returned.
    pub mean_match_similarity: f32,
    /// One-to-one match evidence, ordered by query term index.
    pub matches: Vec<RoleObservationMatch>,
    /// Query role/entity pairs that did not match, in query order.
    pub unmatched_query_terms: Vec<(SemanticRole, String)>,
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
    /// A deterministic maximum-cardinality one-to-one matcher ensures each
    /// query term and stored observation is used at most once. It uses augmenting
    /// paths so a locally highest-scoring pair cannot reduce overall query coverage.
    /// This preserves same-role multiplicity: one stored Agent cannot satisfy two
    /// distinct Agent terms.
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
        self.search_role_observations_with_config(
            query,
            k,
            current_cycle,
            RoleObservationSearchConfig::default(),
        )
    }

    /// Search with an explicit, validated cosine policy so threshold sweeps can
    /// be evaluated without changing code or silently calibrating the default.
    pub fn search_role_observations_with_config(
        &mut self,
        query: &[RoleBoundVector],
        k: usize,
        current_cycle: u64,
        config: RoleObservationSearchConfig,
    ) -> Vec<RoleAwareSearchResult> {
        if query.is_empty() || k == 0 {
            return Vec::new();
        }

        let min_role_match_cosine = config.min_cosine_similarity();
        let mut results = Vec::new();

        for fact in self.facts.values() {
            let observations = &fact.encoding.role_vector_observations;
            if observations.is_empty() {
                continue;
            }

            let mut candidates_by_query = vec![Vec::new(); query.len()];
            for (query_index, query_term) in query.iter().enumerate() {
                for (observation_index, observation) in observations.iter().enumerate() {
                    if query_term.role != observation.role {
                        continue;
                    }
                    let similarity = query_term.vector.cosine_similarity(&observation.vector);
                    if similarity >= min_role_match_cosine {
                        candidates_by_query[query_index].push((observation_index, similarity));
                    }
                }
            }

            // Optimize the assignment lexicographically: maximize covered query terms,
            // then maximize total pairwise similarity among equally complete assignments.
            for candidates in &mut candidates_by_query {
                candidates.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
            }
            let observation_to_query = maximum_weight_maximum_cardinality_matching(
                &candidates_by_query,
                observations.len(),
            );
            let mut assignments: Vec<(usize, usize)> = observation_to_query
                .iter()
                .enumerate()
                .filter_map(|(observation_index, query_index)| {
                    query_index.map(|query_index| (query_index, observation_index))
                })
                .collect();
            assignments.sort_unstable_by_key(|(query_index, _)| *query_index);

            let mut matches = Vec::with_capacity(assignments.len());
            let mut query_matched = vec![false; query.len()];
            let mut total_similarity = 0.0f32;
            for (query_index, observation_index) in assignments {
                if let Some((_, similarity)) = candidates_by_query[query_index]
                    .iter()
                    .find(|(candidate_index, _)| *candidate_index == observation_index)
                {
                    let observation = &observations[observation_index];
                    query_matched[query_index] = true;
                    total_similarity += *similarity;
                    matches.push(RoleObservationMatch {
                        role: query[query_index].role,
                        query_entity_text: query[query_index].entity_text.clone(),
                        observed_entity_text: observation.entity_text.clone(),
                        similarity: *similarity,
                    });
                }
            }

            let matched_query_terms = matches.len();
            if matched_query_terms > 0 {
                let query_term_count = query.len();
                let coverage = matched_query_terms as f32 / query_term_count as f32;
                let mean_match_similarity = total_similarity / matched_query_terms as f32;
                let unmatched_query_terms = query
                    .iter()
                    .zip(query_matched.iter())
                    .filter_map(|(term, matched)| {
                        if *matched {
                            None
                        } else {
                            Some((term.role, term.entity_text.clone()))
                        }
                    })
                    .collect();

                results.push(RoleAwareSearchResult {
                    fact_id: fact.id,
                    similarity: total_similarity / query_term_count as f32,
                    confidence: fact.confidence,
                    matched_query_terms,
                    query_term_count,
                    coverage,
                    mean_match_similarity,
                    matches,
                    unmatched_query_terms,
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

fn maximum_weight_maximum_cardinality_matching(
    candidates_by_query: &[Vec<(usize, f32)>],
    observation_count: usize,
) -> Vec<Option<usize>> {
    #[derive(Clone)]
    struct Edge {
        to: usize,
        reverse_index: usize,
        capacity: u8,
        cost: f64,
    }

    fn add_edge(graph: &mut [Vec<Edge>], from: usize, to: usize, cost: f64) -> usize {
        let forward_index = graph[from].len();
        let reverse_index = graph[to].len();
        graph[from].push(Edge {
            to,
            reverse_index,
            capacity: 1,
            cost,
        });
        graph[to].push(Edge {
            to: from,
            reverse_index: forward_index,
            capacity: 0,
            cost: -cost,
        });
        forward_index
    }

    let query_count = candidates_by_query.len();
    let source = 0;
    let query_start = 1;
    let observation_start = query_start + query_count;
    let sink = observation_start + observation_count;
    let mut residual: Vec<Vec<Edge>> = vec![Vec::new(); sink + 1];
    let mut assignment_edges: Vec<Vec<(usize, usize)>> = vec![Vec::new(); query_count];

    for query_index in 0..query_count {
        add_edge(&mut residual, source, query_start + query_index, 0.0);
    }
    for observation_index in 0..observation_count {
        add_edge(
            &mut residual,
            observation_start + observation_index,
            sink,
            0.0,
        );
    }
    for (query_index, candidates) in candidates_by_query.iter().enumerate() {
        let query_node = query_start + query_index;
        for &(observation_index, similarity) in candidates {
            if observation_index >= observation_count || !similarity.is_finite() {
                continue;
            }
            let observation_node = observation_start + observation_index;
            let edge_index = add_edge(
                &mut residual,
                query_node,
                observation_node,
                -(similarity as f64),
            );
            assignment_edges[query_index].push((observation_index, edge_index));
        }
    }

    loop {
        let node_count = residual.len();
        let mut distance = vec![f64::INFINITY; node_count];
        let mut predecessor: Vec<Option<(usize, usize)>> = vec![None; node_count];
        distance[source] = 0.0;

        // Bellman-Ford over the residual network; insertion order breaks ties.
        for _ in 0..node_count.saturating_sub(1) {
            let mut changed = false;
            for from in 0..node_count {
                if !distance[from].is_finite() {
                    continue;
                }
                for (edge_index, edge) in residual[from].iter().enumerate() {
                    if edge.capacity == 0 {
                        continue;
                    }
                    let candidate_distance = distance[from] + edge.cost;
                    if candidate_distance < distance[edge.to] - 1e-12 {
                        distance[edge.to] = candidate_distance;
                        predecessor[edge.to] = Some((from, edge_index));
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        if predecessor[sink].is_none() {
            break;
        }

        // Augment one unit of flow along the shortest residual path.
        let mut node = sink;
        while node != source {
            let (from, edge_index) = predecessor[node]
                .expect("a reachable augmenting path must have a predecessor");
            let to = residual[from][edge_index].to;
            let reverse_index = residual[from][edge_index].reverse_index;
            residual[from][edge_index].capacity -= 1;
            residual[to][reverse_index].capacity += 1;
            node = from;
        }
    }

    // Return the existing representation: observation index -> assigned query.
    let mut observation_to_query = vec![None; observation_count];
    for (query_index, edges) in assignment_edges.iter().enumerate() {
        let query_node = query_start + query_index;
        for &(observation_index, edge_index) in edges {
            if residual[query_node][edge_index].capacity == 0 {
                observation_to_query[observation_index] = Some(query_index);
                break;
            }
        }
    }
    observation_to_query
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
        assert_eq!(results[0].query_term_count, 2);
        assert!((results[0].coverage - 1.0).abs() < f32::EPSILON);
        assert!((results[0].mean_match_similarity - 1.0).abs() < f32::EPSILON);
        assert!(results[0].unmatched_query_terms.is_empty());
        assert_eq!(results[0].matches.len(), 2);
        assert_eq!(results[0].matches[0].role, SemanticRole::Agent);
        assert_eq!(results[0].matches[0].query_entity_text, "United States");
        assert_eq!(results[0].matches[0].observed_entity_text, "United States");
        assert!((results[0].matches[0].similarity - 1.0).abs() < f32::EPSILON);
        assert_eq!(results[0].matches[1].query_entity_text, "European Union");
        assert_eq!(results[0].matches[1].observed_entity_text, "European Union");
        assert!((results[0].similarity - 1.0).abs() < f32::EPSILON);
        assert!((results[0].confidence - 0.35).abs() < f32::EPSILON);

        let partial_result = results
            .iter()
            .find(|result| result.fact_id == partial_id)
            .expect("one matching Agent should retain partial relevance");
        assert_eq!(partial_result.matched_query_terms, 1);
        assert_eq!(partial_result.query_term_count, 2);
        assert!((partial_result.coverage - 0.5).abs() < f32::EPSILON);
        assert!((partial_result.mean_match_similarity - 1.0).abs() < f32::EPSILON);
        assert_eq!(
            partial_result.unmatched_query_terms,
            vec![(SemanticRole::Agent, "European Union".to_string())]
        );
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
    fn test_role_observation_search_distinguishes_entity_types() {
        use crate::knowledge::encoding::RoleBoundVector;
        use crate::knowledge::extraction::EntityType;

        let mut encoder = KnowledgeEncoder::with_seed(42);
        let place_query = encoder.encode_role_terms(&[(
            SemanticRole::Patient,
            "Mercury",
            EntityType::Place,
        )]);
        let organization_query = encoder.encode_role_terms(&[(
            SemanticRole::Patient,
            "Mercury",
            EntityType::Organization,
        )]);

        assert_eq!(place_query[0].entity_text, organization_query[0].entity_text);
        assert_eq!(place_query[0].role, organization_query[0].role);
        assert!(
            place_query[0].vector.cosine_similarity(&organization_query[0].vector) < 0.1,
            "the same surface text bound to different entity types should be distinguishable"
        );

        let mut graph = EnhancedKnowledgeGraph::new(10);
        let mut place_fact = make_encoding("Mercury the planet", 0.8);
        place_fact.role_vector_observations = vec![RoleBoundVector {
            role: SemanticRole::Patient,
            entity_text: "Mercury".into(),
            vector: place_query[0].vector,
        }];
        let (place_id, _) = graph.insert(place_fact, 1, None, false);

        let mut organization_fact = make_encoding("Mercury the organization", 0.9);
        organization_fact.role_vector_observations = vec![RoleBoundVector {
            role: SemanticRole::Patient,
            entity_text: "Mercury".into(),
            vector: organization_query[0].vector,
        }];
        let (organization_id, _) = graph.insert(organization_fact, 2, None, false);

        let place_results = graph.search_role_observations(&place_query, 10, 3);
        assert_eq!(place_results.len(), 1);
        assert_eq!(place_results[0].fact_id, place_id);
        assert_ne!(place_results[0].fact_id, organization_id);

        let organization_results = graph.search_role_observations(&organization_query, 10, 4);
        assert_eq!(organization_results.len(), 1);
        assert_eq!(organization_results[0].fact_id, organization_id);
        assert_ne!(organization_results[0].fact_id, place_id);
    }

    #[test]
    fn test_role_observation_search_threshold_is_explicit_and_validated() {
        use crate::knowledge::encoding::RoleBoundVector;

        assert!(RoleObservationSearchConfig::new(-1.0).is_ok());
        assert!(RoleObservationSearchConfig::new(1.0).is_ok());
        assert!(RoleObservationSearchConfig::new(1.01).is_err());
        assert!(RoleObservationSearchConfig::new(f32::NAN).is_err());

        let mut encoder = KnowledgeEncoder::with_seed(42);
        let query = encoder.encode_role_terms(&[(
            SemanticRole::Agent,
            "United States",
            EntityType::Organization,
        )]);
        let mut graph = EnhancedKnowledgeGraph::new(10);

        let mut exact = make_encoding("exact typed observation", 0.8);
        exact.role_vector_observations = query.clone();
        let (exact_id, _) = graph.insert(exact, 1, None, false);

        let mut distractor = make_encoding("unrelated same-role observation", 0.8);
        distractor.role_vector_observations = vec![RoleBoundVector {
            role: SemanticRole::Agent,
            entity_text: "Canada".into(),
            vector: BinaryHV::random(123_456),
        }];
        let _ = graph.insert(distractor, 2, None, false);

        let permissive = graph.search_role_observations_with_config(
            &query,
            10,
            3,
            RoleObservationSearchConfig::new(-1.0).unwrap(),
        );
        assert_eq!(permissive.len(), 2);

        let strict = graph.search_role_observations_with_config(
            &query,
            10,
            4,
            RoleObservationSearchConfig::new(1.0).unwrap(),
        );
        assert_eq!(strict.len(), 1);
        assert_eq!(strict[0].fact_id, exact_id);
        assert!((strict[0].coverage - 1.0).abs() < f32::EPSILON);
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

    fn brute_force_matching_objective(
        candidates_by_query: &[Vec<(usize, f32)>],
        observation_count: usize,
    ) -> (usize, f64) {
        fn visit(
            query_index: usize,
            candidates_by_query: &[Vec<(usize, f32)>],
            used_observations: &mut [bool],
            matched_count: usize,
            total_similarity: f64,
            best: &mut (usize, f64),
        ) {
            if query_index == candidates_by_query.len() {
                if matched_count > best.0
                    || (matched_count == best.0 && total_similarity > best.1 + 1e-12)
                {
                    *best = (matched_count, total_similarity);
                }
                return;
            }

            // Leaving a query term unmatched is always a valid candidate.
            visit(
                query_index + 1,
                candidates_by_query,
                used_observations,
                matched_count,
                total_similarity,
                best,
            );

            for &(observation_index, similarity) in &candidates_by_query[query_index] {
                if observation_index >= used_observations.len()
                    || used_observations[observation_index]
                    || !similarity.is_finite()
                {
                    continue;
                }
                used_observations[observation_index] = true;
                visit(
                    query_index + 1,
                    candidates_by_query,
                    used_observations,
                    matched_count + 1,
                    total_similarity + similarity as f64,
                    best,
                );
                used_observations[observation_index] = false;
            }
        }

        let mut best = (0, 0.0);
        visit(
            0,
            candidates_by_query,
            &mut vec![false; observation_count],
            0,
            0.0,
            &mut best,
        );
        best
    }

    #[test]
    fn test_role_matcher_maximizes_similarity_after_coverage() {
        // Both assignments have cardinality two. A first-fit matcher would take
        // query 0 -> observation 0 (0.95) and query 1 -> observation 1 (0.10),
        // total 1.05. The optimal assignment is 0.80 + 0.90 = 1.70.
        let candidates_by_query =
            vec![vec![(0, 0.95), (1, 0.80)], vec![(0, 0.90), (1, 0.10)]];
        let assignment =
            maximum_weight_maximum_cardinality_matching(&candidates_by_query, 2);
        assert_eq!(assignment, vec![Some(1), Some(0)]);
        let total_similarity = 0.80_f32 + 0.90_f32;
        assert!((total_similarity - 1.70).abs() < f32::EPSILON);
    }

    #[test]
    fn test_role_matcher_maximizes_coverage_before_similarity() {
        // A one-edge assignment scores 0.95 but covers only one query term.
        // The selected solution must cover both terms, even at lower total score.
        let candidates_by_query = vec![vec![(0, 0.95), (1, 0.20)], vec![(0, 0.80)]];
        let assignment =
            maximum_weight_maximum_cardinality_matching(&candidates_by_query, 2);
        assert_eq!(assignment, vec![Some(1), Some(0)]);
    }

    #[test]
    fn test_role_matcher_ties_are_deterministic() {
        let candidates_by_query = vec![vec![(0, 0.7), (1, 0.7)], vec![(0, 0.7), (1, 0.7)]];
        let first = maximum_weight_maximum_cardinality_matching(&candidates_by_query, 2);
        let second = maximum_weight_maximum_cardinality_matching(&candidates_by_query, 2);
        assert_eq!(first, second);
        assert_eq!(first.iter().filter(|assigned| assigned.is_some()).count(), 2);
    }

    #[test]
    fn test_role_matcher_ignores_invalid_weights_and_indices() {
        let candidates_by_query = vec![vec![(0, f32::NAN), (7, 0.9), (1, 0.4)]];
        let assignment =
            maximum_weight_maximum_cardinality_matching(&candidates_by_query, 2);
        assert_eq!(assignment, vec![None, Some(0)]);
    }

    #[test]
    fn test_weighted_matcher_agrees_with_exhaustive_oracle_for_all_3x3_graphs() {
        // Enumerate every eligibility graph for three query terms and three
        // observations (2^9 graphs). Compare against an independent exhaustive
        // oracle whose objective is lexicographic: cardinality, then total weight.
        let query_count = 3usize;
        let observation_count = 3usize;
        for edge_mask in 0u16..(1u16 << (query_count * observation_count)) {
            let mut candidates_by_query = vec![Vec::new(); query_count];
            for query_index in 0..query_count {
                for observation_index in 0..observation_count {
                    let bit = query_index * observation_count + observation_index;
                    if edge_mask & (1u16 << bit) != 0 {
                        let weight = (1 + ((query_index * 5 + observation_index * 3) % 9)) as f32
                            / 10.0;
                        candidates_by_query[query_index].push((observation_index, weight));
                    }
                }
            }

            let expected =
                brute_force_matching_objective(&candidates_by_query, observation_count);
            let assignment = maximum_weight_maximum_cardinality_matching(
                &candidates_by_query,
                observation_count,
            );
            let actual_count = assignment
                .iter()
                .filter(|query_index| query_index.is_some())
                .count();
            let actual_weight: f64 = assignment
                .iter()
                .enumerate()
                .filter_map(|(observation_index, query_index)| {
                    let query_index = (*query_index)?;
                    candidates_by_query[query_index]
                        .iter()
                        .find(|(candidate_observation, _)| {
                            *candidate_observation == observation_index
                        })
                        .map(|(_, similarity)| *similarity as f64)
                })
                .sum();

            assert_eq!(
                actual_count, expected.0,
                "weighted matcher has wrong cardinality for edge mask {edge_mask:#05x}"
            );
            assert!(
                (actual_weight - expected.1).abs() < 1e-8,
                "weighted matcher has wrong total score for edge mask {edge_mask:#05x}: got {actual_weight}, expected {}",
                expected.1
            );
        }
    }

    #[test]
    fn test_role_observation_search_labeled_corpus_metrics() {
        use std::collections::HashSet;

        let mut encoder = KnowledgeEncoder::with_seed(42);
        let query_terms = [
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
        ];
        let query = encoder.encode_role_terms(&query_terms);
        assert_eq!(query.len(), 2);

        let unrelated = encoder.encode_role_terms(&[
            (SemanticRole::Agent, "Acme Holdings", EntityType::Organization),
            (SemanticRole::Agent, "Blue Orchard", EntityType::Organization),
        ]);

        let mut graph = EnhancedKnowledgeGraph::new(16);
        let mut positive_ids = HashSet::new();
        let mut negative_ids = HashSet::new();

        let mut exact = make_encoding("corpus/exact-two-agent-positive", 0.35);
        exact.role_vector_observations = query.clone();
        let (exact_id, _) = graph.insert(exact, 1, Some("role-eval".into()), false);
        positive_ids.insert(exact_id);

        let mut partial = make_encoding("corpus/partial-one-agent-positive", 0.99);
        partial.role_vector_observations = vec![query[0].clone()];
        let (partial_id, _) = graph.insert(partial, 2, Some("role-eval".into()), false);
        positive_ids.insert(partial_id);

        let mut wrong_role_observation = query[0].clone();
        wrong_role_observation.role = SemanticRole::Patient;
        let mut wrong_role = make_encoding("corpus/wrong-role-negative", 0.8);
        wrong_role.role_vector_observations = vec![wrong_role_observation];
        let (wrong_role_id, _) = graph.insert(wrong_role, 3, None, false);
        negative_ids.insert(wrong_role_id);

        let mut distractor = make_encoding("corpus/unrelated-same-role-negative", 0.8);
        distractor.role_vector_observations = unrelated;
        let (distractor_id, _) = graph.insert(distractor, 4, None, false);
        negative_ids.insert(distractor_id);

        let wrong_type = encoder.encode_role_terms(&[(
            SemanticRole::Agent,
            "United States",
            EntityType::Place,
        )]);
        let mut same_text_wrong_type =
            make_encoding("corpus/same-text-wrong-type-negative", 0.8);
        same_text_wrong_type.role_vector_observations = wrong_type;
        let (wrong_type_id, _) = graph.insert(same_text_wrong_type, 5, None, false);
        negative_ids.insert(wrong_type_id);

        // Models a pre-multiplicity persisted record: the compatibility map
        // exists, but there are no per-entity observations to score.
        let mut legacy = make_encoding("corpus/legacy-map-only-negative", 0.9);
        legacy
            .role_vectors
            .insert(SemanticRole::Agent, query[0].vector);
        let (legacy_id, _) = graph.insert(legacy, 6, None, false);
        negative_ids.insert(legacy_id);

        let results = graph.search_role_observations(&query, 10, 6);
        let result_ids: Vec<_> = results.iter().map(|result| result.fact_id).collect();
        let result_id_set: HashSet<_> = result_ids.iter().copied().collect();

        let true_positives = positive_ids.intersection(&result_id_set).count();
        let false_positives = negative_ids.intersection(&result_id_set).count();
        let recall_at_10 = true_positives as f32 / positive_ids.len() as f32;

        assert_eq!(positive_ids.len(), 2);
        assert_eq!(recall_at_10, 1.0, "both labeled positives should be retrieved");
        assert_eq!(false_positives, 0, "negative controls must not be retrieved");
        assert_eq!(
            result_ids,
            vec![exact_id, partial_id],
            "full role coverage must outrank partial coverage even with lower confidence"
        );
        assert_eq!(results[0].matched_query_terms, 2);
        assert_eq!(results[1].matched_query_terms, 1);
        assert!(
            results[0].similarity > results[1].similarity,
            "coverage-sensitive retrieval should outrank partial coverage"
        );
        assert!(
            results[0].confidence < results[1].confidence,
            "retrieval ranking must not silently become confidence ranking"
        );

        let top_one = graph.search_role_observations(&query, 1, 7);
        assert_eq!(top_one.len(), 1);
        assert_eq!(top_one[0].fact_id, exact_id);

        // One stored observation cannot be counted twice for repeated query terms.
        let duplicate_term_query = [query[0].clone(), query[0].clone()];
        let duplicate_results =
            graph.search_role_observations(&duplicate_term_query, 10, 8);
        let exact_duplicate_result = duplicate_results
            .iter()
            .find(|result| result.fact_id == exact_id)
            .expect("exact positive should retain one of the repeated terms");
        assert_eq!(exact_duplicate_result.matched_query_terms, 1);
        assert!((exact_duplicate_result.similarity - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_role_observation_search_is_stable_across_sqlite_restart() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_role_search_restart_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");

        let mut encoder = KnowledgeEncoder::with_seed(42);
        let terms = [
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
        ];
        let query = encoder.encode_role_terms(&terms);

        let mut graph = EnhancedKnowledgeGraph::new(10);
        let mut encoding = make_encoding("role-query persistence fixture", 0.73);
        encoding.role_vector_observations = query.clone();
        let (id, _) = graph.insert(encoding, 7, Some("test".into()), false);
        let before = graph.search_role_observations(&query, 5, 8);
        assert_eq!(before.len(), 1);
        assert_eq!(before[0].fact_id, id);

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
        assert_eq!(records.len(), 1);

        let mut restored = EnhancedKnowledgeGraph::new(10);
        assert_eq!(restored.import_fact_records(&records), 1);
        let restored_observations = &restored
            .get_fact(id)
            .expect("restored fact must retain its stable ID")
            .encoding
            .role_vector_observations;
        assert_eq!(restored_observations, &query);

        let after = restored.search_role_observations(&query, 5, 8);
        assert_eq!(after.len(), before.len());
        assert_eq!(after[0].fact_id, before[0].fact_id);
        assert_eq!(after[0].matched_query_terms, before[0].matched_query_terms);
        assert_eq!(after[0].query_term_count, before[0].query_term_count);
        assert_eq!(after[0].coverage, before[0].coverage);
        assert_eq!(after[0].mean_match_similarity, before[0].mean_match_similarity);
        assert_eq!(after[0].matches, before[0].matches);
        assert_eq!(after[0].unmatched_query_terms, before[0].unmatched_query_terms);
        assert!((after[0].similarity - before[0].similarity).abs() < f32::EPSILON);
        assert!((after[0].confidence - before[0].confidence).abs() < f32::EPSILON);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[test]
    fn test_role_matcher_uses_augmenting_paths_to_maximize_coverage() {
        // A greedy global-edge walk would take query 0 -> observation 0 (0.9),
        // leaving query 1 unmatched. Reassignment should find the full solution:
        // query 0 -> observation 1 and query 1 -> observation 0.
        let candidates_by_query = vec![vec![(0, 0.9), (1, 0.2)], vec![(0, 0.8)]];
        let observation_to_query =
            maximum_weight_maximum_cardinality_matching(&candidates_by_query, 2);

        assert_eq!(observation_to_query, vec![Some(1), Some(0)]);
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
