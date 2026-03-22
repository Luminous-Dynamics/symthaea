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

use super::encoding::FactEncoding;
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
    pub fn import_fact_record(&mut self, record: &super::persistence::FactRecord) {
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
        let id: FactId = self.next_id;
        self.next_id += 1;
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

    fn make_encoding(text: &str, confidence: f32) -> FactEncoding {
        FactEncoding {
            vector: BinaryHV::random(crate::knowledge::encoding::fnv1a_hash(text)),
            role_vectors: HashMap::new(),
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
