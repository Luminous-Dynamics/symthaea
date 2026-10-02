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
use symthaea_epistemic_types::{
    CanonicalAdmissionReceipt, MemoryKind, MemoryProvenance, ProvenanceRelation, ProvenanceRelationKind,
    ProvenanceValidationReport, ProvenanceValidationViolation,
    ProvenanceView,
};

// ── Types ──────────────────────────────────────────────────────────────────

/// Unique fact identifier
pub type FactId = u64;

/// Opaque capability representing an explicit canonical-memory admission.
///
/// The fields are intentionally private so retrieval/search code cannot manufacture
/// an admission merely by constructing a provenance envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalAdmission {
    canonical_identity: String,
    provenance_family: Option<String>,
    /// Optional immutable context proving which validated provenance snapshot and
    /// admission event this capability was derived from. Holochain adapters can
    /// require this context without coupling the cognitive core to Holochain types.
    receipt: Option<CanonicalAdmissionReceipt>,
}

impl CanonicalAdmission {
    pub fn new(
        canonical_identity: impl Into<String>,
        provenance_family: Option<String>,
    ) -> Result<Self, &'static str> {
        let canonical_identity = canonical_identity.into();
        if canonical_identity.trim().is_empty() {
            return Err("canonical identity must be non-empty");
        }
        if provenance_family
            .as_deref()
            .is_some_and(|family| family.trim().is_empty())
        {
            return Err("provenance family must be non-empty when present");
        }
        Ok(Self { canonical_identity, provenance_family, receipt: None })
    }

    /// Bind the admission to the immutable provenance snapshot that justified the
    /// software-level admission. The receipt carries no confidence/evidence weight.
    pub fn with_receipt(mut self, receipt: CanonicalAdmissionReceipt) -> Self {
        self.receipt = Some(receipt);
        self
    }

    pub fn receipt(&self) -> Option<&CanonicalAdmissionReceipt> {
        self.receipt.as_ref()
    }
}

/// A fact stored in the knowledge graph with temporal metadata
#[derive(Debug, Clone)]
pub struct TemporalFact {
    /// Stable semantic-memory identity, independent of the in-process FactId.
    pub memory_id: String,
    /// Optional canonical identity assigned by the epistemic admission boundary.
    pub canonical_identity: Option<String>,
    /// Provenance family shared by representations of the same source lineage.
    pub provenance_family: Option<String>,
    /// Unique in-process retrieval identifier
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

/// Result of restoring one persisted fact into the bounded graph.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FactRestoreOutcome {
    /// Whether the persisted fact was admitted into the graph.
    pub accepted: bool,
    /// Number of existing facts evicted by graph retention policy while admitting it.
    pub policy_evictions: usize,
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
    /// Append-only provenance relations between stable memory identities.
    /// These relations are structural lineage, not evidence-weighting signals.
    provenance_relations: Vec<ProvenanceRelation>,
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
            provenance_relations: Vec::new(),
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

        // Never collapse a new observation into an existing fact based on HDC similarity.
        // Similarity is retrieval metadata, not provenance or independent evidence.
        // Explicit corroboration is represented by ProvenanceRelationKind::Corroborates
        // and must therefore be recorded at the provenance boundary.
        // Evict if at capacity
        if self.facts.len() >= self.capacity {
            self.evict_lowest_confidence();
        }

        let id = self.next_id;
        self.next_id += 1;
        let memory_id = uuid::Uuid::new_v4().to_string();

        let confidence = encoding.confidence;
        let fact = TemporalFact {
            memory_id,
            canonical_identity: None,
            provenance_family: None,
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
                .then_with(|| a.fact_id.cmp(&b.fact_id))
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
                .then_with(|| a.fact_id.cmp(&b.fact_id))
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

    /// Return provenance metadata without treating retrieval metrics as evidence.
    pub fn provenance(&self, id: FactId) -> Option<MemoryProvenance> {
        self.facts.get(&id).map(|fact| MemoryProvenance {
            canonical_identity: fact.canonical_identity.clone(),
            memory_id: fact.memory_id.clone(),
            memory_kind: MemoryKind::KnowledgeGraph,
            created_at: format!("cycle:{}", fact.inserted_at_cycle),
            source_event: None,
            canonical_artifact_ref: None,
            statement_ref: None,
            provenance_family: fact.provenance_family.clone(),
            epistemic_state: None,
            claim_ceiling: None,
            frontier_ref: None,
            derivation_ref: None,
            model_ref: None,
            retrieval_index_ref: Some(format!("fact-id:{}", fact.id)),
        })
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

    /// Admit a canonical identity into an opaque capability.
    /// This is the sole construction path for the capability accepted by
    /// attach_admitted_provenance; retrieval and HDC similarity do not create it.
    pub fn admit_canonical_identity(
        &self,
        canonical_identity: impl Into<String>,
        provenance_family: Option<String>,
    ) -> Result<CanonicalAdmission, &'static str> {
        CanonicalAdmission::new(canonical_identity, provenance_family)
    }

    /// Attach an explicitly admitted canonical/provenance identity to a local fact.
    /// The retrieval handle remains unchanged. No confidence or evidence score is modified.
    pub fn attach_admitted_provenance(
        &mut self,
        id: FactId,
        admission: CanonicalAdmission,
    ) -> bool {
        if let Some(receipt) = admission.receipt() {
            // A receipt is only meaningful if it binds this graph's exact current
            // structural provenance snapshot. Reject stale or non-conforming admission
            // context rather than allowing a canonical identity to bypass the boundary.
            let validation = self.validate_provenance();
            if !receipt.binds_validation(&validation) {
                return false;
            }
        }

        if let Some(fact) = self.facts.get_mut(&id) {
            fact.canonical_identity = Some(admission.canonical_identity);
            fact.provenance_family = admission.provenance_family;
            true
        } else {
            false
        }
    }

    /// Record a typed provenance relation without changing either endpoint's confidence.
    pub fn record_provenance_relation(&mut self, relation: ProvenanceRelation) -> Result<bool, &'static str> {
        relation.validate()?;
        // Endpoint memories may have been evicted from the local cognitive projection;
        // provenance history must remain referentially stable rather than being erased by eviction.
        if matches!(relation.kind, ProvenanceRelationKind::DerivedFrom | ProvenanceRelationKind::RevisedFrom | ProvenanceRelationKind::Supersedes)
            && self.lineage_would_cycle(&relation.source_memory_id, &relation.target_memory_id)
        {
            return Err("derivation lineage relation would create a cycle");
        }
        if self.provenance_relations.iter().any(|existing| existing == &relation) {
            return Ok(false);
        }
        self.provenance_relations.push(relation);
        Ok(true)
    }

    pub fn provenance_relations(&self) -> &[ProvenanceRelation] {
        &self.provenance_relations
    }

    /// Export a read-only provenance boundary for evidence/federation adapters.
    ///
    /// The returned view is a snapshot: mutating the cognitive graph afterwards cannot
    /// mutate the view. The view carries lineage plus structural validation only.
    pub fn provenance_view(&self) -> ProvenanceView {
        ProvenanceView::from_relations(
            &self.provenance_relations,
            self.validate_provenance(),
        )
        .expect("graph provenance view must match its validation snapshot")
    }

    /// Validate the current provenance snapshot without mutating graph state.
    ///
    /// The report is structural only: it binds to an order-independent snapshot
    /// digest and checks relation well-formedness plus derivation/revision/supersession acyclicity.
    /// It does not assign truth, reliability, or evidential weight.
    pub fn validate_provenance(&self) -> ProvenanceValidationReport {
        let mut violations = Vec::new();

        for relation in &self.provenance_relations {
            if let Err(message) = relation.validate() {
                violations.push(ProvenanceValidationViolation {
                    code: "invalid_relation".into(),
                    source_memory_id: Some(relation.source_memory_id.clone()),
                    target_memory_id: Some(relation.target_memory_id.clone()),
                    message: message.into(),
                });
            }
        }

        let mut seen = std::collections::HashSet::new();
        for relation in &self.provenance_relations {
            if !seen.insert(relation) {
                violations.push(ProvenanceValidationViolation {
                    code: "duplicate_relation".into(),
                    source_memory_id: Some(relation.source_memory_id.clone()),
                    target_memory_id: Some(relation.target_memory_id.clone()),
                    message: "duplicate provenance relation".into(),
                });
            }
        }

        let lineage: Vec<&ProvenanceRelation> = self
            .provenance_relations
            .iter()
            .filter(|relation| {
                matches!(
                    relation.kind,
                    ProvenanceRelationKind::DerivedFrom
                        | ProvenanceRelationKind::RevisedFrom
                        | ProvenanceRelationKind::Supersedes
                )
            })
            .collect();

        for relation in &lineage {
            let mut frontier = vec![relation.target_memory_id.clone()];
            let mut visited = std::collections::HashSet::new();
            while let Some(current) = frontier.pop() {
                if current == relation.source_memory_id {
                    violations.push(ProvenanceValidationViolation {
                        code: "lineage_cycle".into(),
                        source_memory_id: Some(relation.source_memory_id.clone()),
                        target_memory_id: Some(relation.target_memory_id.clone()),
                        message: "derivation/revision/supersession lineage contains a cycle".into(),
                    });
                    break;
                }
                if !visited.insert(current.clone()) {
                    continue;
                }
                for edge in &lineage {
                    if edge.source_memory_id == current {
                        frontier.push(edge.target_memory_id.clone());
                    }
                }
            }
        }

        ProvenanceValidationReport::from_relations(&self.provenance_relations)
            .with_violations(violations)
    }

    /// Query provenance without assigning evidential weight.
    pub fn provenance_relations_from(&self, memory_id: &str) -> Vec<&ProvenanceRelation> {
        self.provenance_relations
            .iter()
            .filter(|relation| relation.source_memory_id == memory_id)
            .collect()
    }

    /// Query incoming provenance without assigning evidential weight.
    pub fn provenance_relations_to(&self, memory_id: &str) -> Vec<&ProvenanceRelation> {
        self.provenance_relations
            .iter()
            .filter(|relation| relation.target_memory_id == memory_id)
            .collect()
    }

    /// Query one provenance relation kind without assigning evidential weight.
    pub fn provenance_relations_of_kind(
        &self,
        kind: ProvenanceRelationKind,
    ) -> Vec<&ProvenanceRelation> {
        self.provenance_relations
            .iter()
            .filter(|relation| relation.kind == kind)
            .collect()
    }

    pub fn import_provenance_relation(&mut self, relation: ProvenanceRelation) -> Result<bool, &'static str> {
        self.record_provenance_relation(relation)
    }

    fn lineage_would_cycle(&self, source: &str, target: &str) -> bool {
        let mut frontier = vec![target.to_owned()];
        let mut visited = std::collections::HashSet::new();
        while let Some(current) = frontier.pop() {
            if current == source { return true; }
            if !visited.insert(current.clone()) { continue; }
            for relation in &self.provenance_relations {
                if matches!(relation.kind, ProvenanceRelationKind::DerivedFrom | ProvenanceRelationKind::RevisedFrom | ProvenanceRelationKind::Supersedes)
                    && relation.source_memory_id == current
                {
                    frontier.push(relation.target_memory_id.clone());
                }
            }
        }
        false
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

        // HashMap iteration is intentionally arbitrary in Rust. Do not let the
        // discovery order decide which parent first activates a node: different
        // parents can yield different activation values. Instead, collect every
        // candidate at a hop and retain the strongest activation for each node.
        // This makes the traversal path-independent and reproducible.
        let mut facts: Vec<&TemporalFact> = self.facts.values().collect();
        facts.sort_by_key(|fact| fact.id);

        for hop in 0..hops {
            let hop_decay = decay_factor.powi(hop as i32 + 1);
            let mut candidates: HashMap<FactId, f32> = HashMap::new();

            frontier_ids.sort_unstable();

            for &fid in &frontier_ids {
                let query_vec = match self.facts.get(&fid) {
                    Some(f) => f.encoding.vector.clone(),
                    None => continue,
                };

                for fact in &facts {
                    if activated.contains_key(&fact.id) {
                        continue;
                    }
                    let sim = fact.encoding.vector.similarity(&query_vec);
                    if sim > 0.1 {
                        let decayed_sim = sim * hop_decay;
                        candidates
                            .entry(fact.id)
                            .and_modify(|existing| *existing = existing.max(decayed_sim))
                            .or_insert(decayed_sim);
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            frontier_ids = candidates.keys().copied().collect();
            frontier_ids.sort_unstable();

            for (id, activation) in candidates {
                activated.insert(id, activation);
            }
        }

        // Remove seeds from results (caller already has them)
        let seed_ids: std::collections::HashSet<FactId> =
            seeds.iter().map(|s| s.fact_id).collect();

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
                .then_with(|| a.fact_id.cmp(&b.fact_id))
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

    /// Import a fact from a persistence record, exposing retention-policy outcomes.
    pub fn import_fact_record_with_outcome(
        &mut self,
        record: &super::persistence::FactRecord,
    ) -> FactRestoreOutcome {
        if record.vector_bytes.len() != 2048 || self.capacity == 0 {
            return FactRestoreOutcome::default();
        }
        let mut arr = [0u8; 2048];
        arr.copy_from_slice(&record.vector_bytes);
        let encoding = super::encoding::FactEncoding {
            vector: symthaea_core::hdc::binary_hv::BinaryHV(arr),
            role_vectors: std::collections::HashMap::new(),
            source_text: record.source_text.clone(),
            confidence: record.confidence,
        };

        let mut policy_evictions = 0;
        if self.facts.len() >= self.capacity && self.evict_lowest_confidence() {
            policy_evictions = 1;
        }

        let id: FactId = self.next_id;
        self.next_id += 1;
        let fact = TemporalFact {
            memory_id: record.memory_id.clone(),
            canonical_identity: record.canonical_identity.clone(),
            provenance_family: record.provenance_family.clone(),
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
        FactRestoreOutcome {
            accepted: true,
            policy_evictions,
        }
    }

    /// Import a fact from a persistence record.
    pub fn import_fact_record(&mut self, record: &super::persistence::FactRecord) -> bool {
        self.import_fact_record_with_outcome(record).accepted
    }

    /// Export all facts as persistence records.
    pub fn export_fact_records(&self) -> Vec<super::persistence::FactRecord> {
        let mut facts: Vec<&TemporalFact> = self.facts.values().collect();
        facts.sort_by(|a, b| {
            a.memory_id
                .cmp(&b.memory_id)
                .then_with(|| a.id.cmp(&b.id))
        });
        facts
            .into_iter()
            .map(|f| super::persistence::FactRecord {
                memory_id: f.memory_id.clone(),
                canonical_identity: f.canonical_identity.clone(),
                provenance_family: f.provenance_family.clone(),
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

    fn evict_lowest_confidence(&mut self) -> bool {
        if let Some((&id, _)) = self.facts.iter().min_by(|(_, a), (_, b)| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.memory_id.cmp(&b.memory_id))
                .then_with(|| a.id.cmp(&b.id))
        }) {
            self.remove_fact(id);
            self.total_evictions += 1;
            true
        } else {
            false
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
    fn test_similar_insertions_preserve_distinct_memory_identity() {
        let mut graph = EnhancedKnowledgeGraph::new(100);

        let enc1 = make_encoding("oil prices rose", 0.7);
        let (id1, _) = graph.insert(enc1, 1, None, false);

        // A second observation is a distinct memory until an explicit provenance
        // relation says otherwise. HDC similarity must not silently collapse it.
        let enc2 = make_encoding("oil prices rose", 0.8);
        let (id2, _) = graph.insert(enc2, 2, None, false);

        assert_ne!(id1, id2);
        assert_eq!(graph.len(), 2);
        assert_eq!(graph.get_fact(id1).unwrap().corroboration_count, 0);
        assert_eq!(graph.get_fact(id2).unwrap().corroboration_count, 0);

        let memory_1 = graph.provenance(id1).unwrap().memory_id;
        let memory_2 = graph.provenance(id2).unwrap().memory_id;
        assert_ne!(memory_1, memory_2);

        graph.record_provenance_relation(ProvenanceRelation {
            source_memory_id: memory_2,
            target_memory_id: memory_1,
            kind: ProvenanceRelationKind::Corroborates,
            created_at: "cycle:2".into(),
        }).unwrap();

        // The relation is structural provenance, not an implicit confidence boost.
        assert_eq!(graph.get_fact(id1).unwrap().confidence, 0.7);
        assert_eq!(graph.get_fact(id2).unwrap().confidence, 0.8);
    }

    #[test]
    fn test_retrieval_repetition_does_not_create_corroboration() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let (id, _) = graph.insert(make_encoding("retrieval-only claim", 0.8), 1, None, false);
        let query = graph.get_fact(id).unwrap().encoding.vector.clone();

        for cycle in 2..=10 {
            let _ = graph.search(&query, 5, cycle);
        }

        let fact = graph.get_fact(id).unwrap();
        assert_eq!(fact.corroboration_count, 0);
        assert!((fact.confidence - 0.8).abs() < f32::EPSILON);
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
    fn test_memory_identity_survives_export_import() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let enc = make_encoding("stable claim", 0.8);
        let (id, _) = graph.insert(enc, 1, Some("test".into()), false);

        let original = graph.provenance(id).unwrap();
        let records = graph.export_fact_records();

        let mut restored = EnhancedKnowledgeGraph::new(100);
        restored.import_fact_record(&records[0]);
        let restored_record = &restored.export_fact_records()[0];

        assert_eq!(restored_record.memory_id, original.memory_id);
        assert_eq!(restored_record.provenance_family, original.provenance_family);
        assert_eq!(restored_record.source_text, "stable claim");

        let restored_id = restored.facts.keys().next().copied().unwrap();
        let round_trip = restored.provenance(restored_id).unwrap();
        assert_eq!(round_trip.retrieval_index_ref, Some(format!("fact-id:{restored_id}")));
        assert_ne!(restored_id, id);
    }

    #[test]
    fn test_graph_search_and_export_have_stable_tie_order() {
        let mut graph = EnhancedKnowledgeGraph::new(10);
        let vector = make_encoding("same-vector", 0.8).vector;
        for memory_id in ["memory-b", "memory-a"] {
            graph.import_fact_record(&super::super::persistence::FactRecord {
                memory_id: memory_id.into(),
                canonical_identity: None,
                provenance_family: None,
                vector_bytes: vector.0.to_vec(),
                source_text: memory_id.into(),
                confidence: 0.8,
                domain: None,
                cycle: 1,
                is_causal: false,
            });
        }

        let results = graph.search(&vector, 2, 2);
        assert_eq!(
            results.iter().map(|result| result.fact_id).collect::<Vec<_>>(),
            vec![1, 2]
        );

        let exported = graph.export_fact_records();
        assert_eq!(
            exported.iter().map(|record| record.memory_id.as_str()).collect::<Vec<_>>(),
            vec!["memory-a", "memory-b"]
        );
    }

    #[test]
    fn test_fact_restore_enforces_capacity_and_reports_policy_eviction() {
        let mut graph = EnhancedKnowledgeGraph::new(2);
        let records = [
            super::super::persistence::FactRecord {
                memory_id: "memory-low".into(),
                canonical_identity: None,
                provenance_family: None,
                vector_bytes: vec![0u8; 2048],
                source_text: "low".into(),
                confidence: 0.2,
                domain: Some("test".into()),
                cycle: 1,
                is_causal: false,
            },
            super::super::persistence::FactRecord {
                memory_id: "memory-high".into(),
                canonical_identity: None,
                provenance_family: None,
                vector_bytes: vec![1u8; 2048],
                source_text: "high".into(),
                confidence: 0.9,
                domain: Some("test".into()),
                cycle: 2,
                is_causal: false,
            },
            super::super::persistence::FactRecord {
                memory_id: "memory-mid".into(),
                canonical_identity: None,
                provenance_family: None,
                vector_bytes: vec![2u8; 2048],
                source_text: "mid".into(),
                confidence: 0.8,
                domain: Some("test".into()),
                cycle: 3,
                is_causal: false,
            },
        ];

        assert!(graph.import_fact_record_with_outcome(&records[0]).accepted);
        assert!(graph.import_fact_record_with_outcome(&records[1]).accepted);
        let outcome = graph.import_fact_record_with_outcome(&records[2]);

        assert_eq!(outcome.policy_evictions, 1);
        assert_eq!(graph.len(), 2);
        assert_eq!(graph.total_evictions(), 1);
        assert!(graph.all_facts().all(|fact| fact.memory_id != "memory-low"));
        assert_eq!(graph.domain_count(), 1);
    }

    #[test]
    fn test_fact_restore_eviction_tie_break_is_deterministic() {
        let mut graph = EnhancedKnowledgeGraph::new(2);
        let first = super::super::persistence::FactRecord {
            memory_id: "memory-z".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0u8; 2048],
            source_text: "z".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        let second = super::super::persistence::FactRecord {
            memory_id: "memory-a".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![1u8; 2048],
            source_text: "a".into(),
            confidence: 0.5,
            domain: None,
            cycle: 2,
            is_causal: false,
        };
        let third = super::super::persistence::FactRecord {
            memory_id: "memory-m".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![2u8; 2048],
            source_text: "m".into(),
            confidence: 0.5,
            domain: None,
            cycle: 3,
            is_causal: false,
        };

        assert!(graph.import_fact_record_with_outcome(&first).accepted);
        assert!(graph.import_fact_record_with_outcome(&second).accepted);
        let outcome = graph.import_fact_record_with_outcome(&third);

        assert_eq!(outcome.policy_evictions, 1);
        assert_eq!(
            graph.all_facts().map(|fact| fact.memory_id.as_str()).collect::<Vec<_>>(),
            vec!["memory-z", "memory-m"]
        );
    }

    #[test]
    fn test_provenance_family_does_not_collapse_memory_identity() {
        let mut graph = EnhancedKnowledgeGraph::new(100);

        for (memory_id, family, text) in [
            ("memory-a", "source-family-1", "representation a"),
            ("memory-b", "source-family-1", "representation b"),
            ("memory-c", "source-family-2", "independent source"),
        ] {
            let encoding = make_encoding(text, 0.8);
            graph.import_fact_record(&super::super::persistence::FactRecord {
                memory_id: memory_id.into(),
                canonical_identity: None,
                provenance_family: Some(family.into()),
                vector_bytes: encoding.vector.0.to_vec(),
                source_text: text.into(),
                confidence: 0.8,
                domain: None,
                cycle: 1,
                is_causal: false,
            });
        }

        let records = graph.export_fact_records();
        assert_eq!(records.len(), 3);

        let same_family: Vec<_> = records
            .iter()
            .filter(|r| r.provenance_family.as_deref() == Some("source-family-1"))
            .collect();
        assert_eq!(same_family.len(), 2);
        assert_ne!(same_family[0].memory_id, same_family[1].memory_id);

        let independent: Vec<_> = records
            .iter()
            .filter(|r| r.provenance_family.as_deref() == Some("source-family-2"))
            .collect();
        assert_eq!(independent.len(), 1);
        assert_ne!(same_family[0].memory_id, independent[0].memory_id);
    }

    #[test]
    fn test_admission_receipt_must_bind_current_conforming_snapshot() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let (id, _) = graph.insert(make_encoding("receipt-bound fact", 0.8), 1, None, false);

        let validation = graph.validate_provenance();
        let receipt = CanonicalAdmissionReceipt::new(
            "admission:event-1",
            Some("frontier:1".into()),
            validation.snapshot_digest.clone(),
            validation.validator_version.clone(),
            validation.snapshot_schema_version,
        ).unwrap();
        let admission = graph
            .admit_canonical_identity("claim:receipt-bound", Some("family:1".into()))
            .unwrap()
            .with_receipt(receipt);
        assert!(graph.attach_admitted_provenance(id, admission));
        assert_eq!(
            graph.get_fact(id).unwrap().canonical_identity.as_deref(),
            Some("claim:receipt-bound")
        );

        let (id2, _) = graph.insert(make_encoding("stale receipt fact", 0.8), 2, None, false);
        let stale = graph.validate_provenance();
        let stale_receipt = CanonicalAdmissionReceipt::new(
            "admission:event-stale",
            Some("frontier:1".into()),
            stale.snapshot_digest,
            stale.validator_version,
            stale.snapshot_schema_version,
        ).unwrap();

        graph.record_provenance_relation(ProvenanceRelation {
            source_memory_id: graph.provenance(id2).unwrap().memory_id,
            target_memory_id: graph.provenance(id).unwrap().memory_id,
            kind: ProvenanceRelationKind::Corroborates,
            created_at: "cycle:3".into(),
        }).unwrap();

        let stale_admission = graph
            .admit_canonical_identity("claim:must-not-bypass", Some("family:2".into()))
            .unwrap()
            .with_receipt(stale_receipt);
        assert!(!graph.attach_admitted_provenance(id2, stale_admission));
        assert!(graph.get_fact(id2).unwrap().canonical_identity.is_none());
    }

    #[test]
    fn test_canonical_identity_requires_explicit_admission() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let enc = make_encoding("admission boundary claim", 0.8);
        let (id, _) = graph.insert(enc, 1, None, false);

        let before = graph.provenance(id).unwrap();
        assert!(before.canonical_identity.is_none());

        let query = graph.get_fact(id).unwrap().encoding.vector.clone();
        let _ = graph.search(&query, 1, 2);
        assert!(graph.provenance(id).unwrap().canonical_identity.is_none());

        let admission = graph
            .admit_canonical_identity("canonical:claim-1", Some("source-family-1".into()))
            .unwrap();
        assert!(graph.attach_admitted_provenance(id, admission));
        let after = graph.provenance(id).unwrap();
        assert_eq!(after.canonical_identity.as_deref(), Some("canonical:claim-1"));
        assert_eq!(after.provenance_family.as_deref(), Some("source-family-1"));
    }

    #[test]
    fn canonical_admission_receipt_binds_snapshot_context_without_evidence_weight() {
        let graph = EnhancedKnowledgeGraph::new(100);
        let receipt = CanonicalAdmissionReceipt::new(
            "admission:event-1",
            Some("frontier:7".into()),
            "digest-abc",
            "melothaea-provenance-structural-v1",
            1,
        )
        .unwrap();
        let admission = graph
            .admit_canonical_identity("canonical:claim-1", Some("family-1".into()))
            .unwrap()
            .with_receipt(receipt.clone());
        assert_eq!(admission.receipt(), Some(&receipt));
        assert_eq!(admission.canonical_identity, "canonical:claim-1");
        assert_eq!(admission.provenance_family.as_deref(), Some("family-1"));
    }

    #[test]
    fn canonical_admission_receipt_rejects_missing_context() {
        assert_eq!(
            CanonicalAdmissionReceipt::new(" ", None, "digest", "validator", 1).unwrap_err(),
            "admission event must be non-empty"
        );
        assert_eq!(
            CanonicalAdmissionReceipt::new("event", None, " ", "validator", 1).unwrap_err(),
            "provenance snapshot digest must be non-empty"
        );
    }

    #[test]
    fn canonical_admission_rejects_empty_identity() {
        let graph = EnhancedKnowledgeGraph::new(100);
        assert_eq!(graph.admit_canonical_identity("  ", None).unwrap_err(), "canonical identity must be non-empty");
        assert_eq!(graph.admit_canonical_identity("canonical:1", Some(" ".into())).unwrap_err(), "provenance family must be non-empty when present");
    }

    #[test]
    fn retrieval_cannot_supply_canonical_admission_capability() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let (id, _) = graph.insert(make_encoding("retrieval-only", 0.8), 1, None, false);
        let query = graph.get_fact(id).unwrap().encoding.vector.clone();
        let _ = graph.search(&query, 1, 2);
        assert!(graph.provenance(id).unwrap().canonical_identity.is_none());
    }

    #[test]
    fn test_eviction_does_not_reassign_surviving_identity() {
        let mut graph = EnhancedKnowledgeGraph::new(2);
        let (id_a, _) = graph.insert(make_encoding("claim a", 0.9), 1, None, false);
        graph.insert(make_encoding("claim b", 0.8), 2, None, false);
        let memory_a = graph.provenance(id_a).unwrap().memory_id;

        graph.insert(make_encoding("claim c", 0.7), 3, None, false);

        assert_eq!(graph.provenance(id_a).unwrap().memory_id, memory_a);
    }

    #[test]
    fn test_provenance_view_is_snapshot_and_non_mutating() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let (a, _) = graph.insert(make_encoding("source", 0.8), 1, None, false);
        let (b, _) = graph.insert(make_encoding("derived", 0.6), 2, None, false);
        let source = graph.provenance(a).unwrap().memory_id;
        let derived = graph.provenance(b).unwrap().memory_id;

        graph.record_provenance_relation(ProvenanceRelation {
            source_memory_id: derived.clone(),
            target_memory_id: source.clone(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        }).unwrap();

        let view = graph.provenance_view();
        assert_eq!(view.relations_from(&derived).len(), 1);
        assert!(view.is_structurally_conforming());

        graph.record_provenance_relation(ProvenanceRelation {
            source_memory_id: source,
            target_memory_id: derived,
            kind: ProvenanceRelationKind::Corroborates,
            created_at: "cycle:3".into(),
        }).unwrap();

        assert_eq!(view.relations.len(), 1);
        assert_eq!(graph.provenance_view().relations.len(), 2);
    }

    #[test]
    fn test_provenance_relation_is_append_only_and_non_evidence_weighting() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let (a, _) = graph.insert(make_encoding("source", 0.8), 1, None, false);
        let (b, _) = graph.insert(make_encoding("derived", 0.6), 2, None, false);
        let source_id = graph.provenance(a).unwrap().memory_id;
        let derived_id = graph.provenance(b).unwrap().memory_id;
        let before = graph.get_fact(b).unwrap().confidence;
        let relation = ProvenanceRelation { source_memory_id: derived_id, target_memory_id: source_id, kind: ProvenanceRelationKind::DerivedFrom, created_at: "cycle:2".into() };
        assert_eq!(graph.record_provenance_relation(relation.clone()).unwrap(), true);
        assert_eq!(graph.record_provenance_relation(relation).unwrap(), false);
        assert_eq!(graph.provenance_relations().len(), 1);
        assert_eq!(graph.get_fact(b).unwrap().confidence, before);

        let corroborates = ProvenanceRelation {
            source_memory_id: graph.provenance(a).unwrap().memory_id,
            target_memory_id: graph.provenance(b).unwrap().memory_id,
            kind: ProvenanceRelationKind::Corroborates,
            created_at: "cycle:2".into(),
        };
        assert!(graph.record_provenance_relation(corroborates).unwrap());
        assert_eq!(graph.get_fact(b).unwrap().confidence, before);

        let representation = ProvenanceRelation {
            source_memory_id: graph.provenance(b).unwrap().memory_id,
            target_memory_id: graph.provenance(a).unwrap().memory_id,
            kind: ProvenanceRelationKind::RepresentationOf,
            created_at: "cycle:2".into(),
        };
        assert!(graph.record_provenance_relation(representation).unwrap());
        assert_eq!(graph.get_fact(b).unwrap().confidence, before);
    }

    #[test]
    fn test_provenance_relation_survives_local_eviction_boundary() {
        let mut graph = EnhancedKnowledgeGraph::new(1);
        let (a, _) = graph.insert(make_encoding("source", 0.8), 1, None, false);
        let a_id = graph.provenance(a).unwrap().memory_id;
        let (b, _) = graph.insert(make_encoding("replacement", 0.7), 2, None, false);
        let b_id = graph.provenance(b).unwrap().memory_id;

        assert!(graph.provenance(a).is_none());
        let relation = ProvenanceRelation {
            source_memory_id: b_id,
            target_memory_id: a_id.clone(),
            kind: ProvenanceRelationKind::RevisedFrom,
            created_at: "cycle:2".into(),
        };
        assert!(graph.record_provenance_relation(relation).is_ok());
        assert_eq!(graph.provenance_relations()[0].target_memory_id, a_id);
    }

    #[test]
    fn test_provenance_queries_are_non_mutating() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let (a, _) = graph.insert(make_encoding("a", 0.8), 1, None, false);
        let (b, _) = graph.insert(make_encoding("b", 0.6), 2, None, false);
        let a_id = graph.provenance(a).unwrap().memory_id;
        let b_id = graph.provenance(b).unwrap().memory_id;
        let before = graph.get_fact(b).unwrap().confidence;
        graph.record_provenance_relation(ProvenanceRelation {
            source_memory_id: b_id.clone(),
            target_memory_id: a_id.clone(),
            kind: ProvenanceRelationKind::Corroborates,
            created_at: "cycle:2".into(),
        }).unwrap();
        assert_eq!(graph.provenance_relations_from(&b_id).len(), 1);
        assert_eq!(graph.provenance_relations_to(&a_id).len(), 1);
        assert_eq!(graph.provenance_relations_of_kind(ProvenanceRelationKind::Corroborates).len(), 1);
        assert_eq!(graph.get_fact(b).unwrap().confidence, before);
    }

    #[test]
    fn test_provenance_validation_report_is_non_mutating_and_snapshot_bound() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let (a, _) = graph.insert(make_encoding("a", 0.8), 1, None, false);
        let (b, _) = graph.insert(make_encoding("b", 0.8), 2, None, false);
        let a_id = graph.provenance(a).unwrap().memory_id;
        let b_id = graph.provenance(b).unwrap().memory_id;

        graph.record_provenance_relation(ProvenanceRelation {
            source_memory_id: b_id,
            target_memory_id: a_id,
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        }).unwrap();

        let before = graph.provenance_relations().to_vec();
        let report = graph.validate_provenance();
        assert!(report.conforms);
        assert_eq!(report.relation_count, 1);
        assert!(!report.snapshot_digest.is_empty());
        assert_eq!(graph.provenance_relations(), before.as_slice());
    }

    #[test]
    fn test_provenance_lineage_rejects_cycles() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let (a, _) = graph.insert(make_encoding("a", 0.8), 1, None, false);
        let (b, _) = graph.insert(make_encoding("b", 0.8), 2, None, false);
        let a_id = graph.provenance(a).unwrap().memory_id;
        let b_id = graph.provenance(b).unwrap().memory_id;
        graph.record_provenance_relation(ProvenanceRelation { source_memory_id: b_id.clone(), target_memory_id: a_id.clone(), kind: ProvenanceRelationKind::DerivedFrom, created_at: "cycle:2".into() }).unwrap();
        let err = graph.record_provenance_relation(ProvenanceRelation { source_memory_id: a_id, target_memory_id: b_id, kind: ProvenanceRelationKind::RevisedFrom, created_at: "cycle:3".into() }).unwrap_err();
        assert_eq!(err, "derivation lineage relation would create a cycle");
    }

    #[test]
    fn test_supersession_lineage_rejects_cycles() {
        let mut graph = EnhancedKnowledgeGraph::new(100);
        let (a, _) = graph.insert(make_encoding("a", 0.8), 1, None, false);
        let (b, _) = graph.insert(make_encoding("b", 0.8), 2, None, false);
        let a_id = graph.provenance(a).unwrap().memory_id;
        let b_id = graph.provenance(b).unwrap().memory_id;

        graph.record_provenance_relation(ProvenanceRelation {
            source_memory_id: b_id.clone(),
            target_memory_id: a_id.clone(),
            kind: ProvenanceRelationKind::Supersedes,
            created_at: "cycle:2".into(),
        }).unwrap();

        let err = graph.record_provenance_relation(ProvenanceRelation {
            source_memory_id: a_id,
            target_memory_id: b_id,
            kind: ProvenanceRelationKind::Supersedes,
            created_at: "cycle:3".into(),
        }).unwrap_err();
        assert_eq!(err, "derivation lineage relation would create a cycle");
    }

    #[test]
    fn test_empty_graph() {
        let graph = EnhancedKnowledgeGraph::new(100);
        assert!(graph.is_empty());
        assert_eq!(graph.average_confidence(), 0.0);
        assert_eq!(graph.domain_count(), 0);
    }
}
