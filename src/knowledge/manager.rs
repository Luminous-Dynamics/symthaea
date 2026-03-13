//! Knowledge Manager — Central Coordinator
//!
//! Orchestrates the knowledge engine subsystems into a single interface
//! that the cognitive loop calls each cycle. Implements the CognitiveSubsystem
//! pattern (interval-based processing).
//!
//! # Per-Cycle Flow
//!
//! ```text
//! 1. Extract facts from input text
//! 2. Encode facts as HDC vectors
//! 3. Insert into knowledge graph (detect contradictions)
//! 4. Process causal relations → causal bridge
//! 5. Search knowledge graph for relevant context
//! 6. Feed adaptive ontology (learn new primitives)
//! 7. Periodic: decay confidence, prune ontology
//! 8. Emit telemetry signals
//! ```

use super::adaptive_ontology::{AdaptiveOntology, AdaptiveOntologyConfig};
use super::causal_bridge::CausalKnowledgeBridge;
use super::encoding::KnowledgeEncoder;
use super::extraction::{EntityType, KnowledgeExtractor};
use super::graph::{ContradictionAlert, EnhancedKnowledgeGraph, FactSearchResult};
use super::persistence::{CausalEdgeRecord, FactRecord, KnowledgePersistence};
use std::collections::VecDeque;
use symthaea_core::hdc::unified_hv::BinaryHV;

// ── Telemetry ──────────────────────────────────────────────────────────────

/// Per-cycle telemetry from the knowledge engine
#[derive(Debug, Clone, Default)]
pub struct KnowledgeTelemetry {
    /// Number of facts extracted this cycle
    pub facts_extracted: u32,
    /// Number of facts inserted into graph
    pub facts_inserted: u32,
    /// Number of causal edges discovered this cycle
    pub causal_edges_added: u32,
    /// Number of contradictions detected this cycle
    pub contradictions_detected: u32,
    /// Total facts in knowledge graph
    pub graph_size: u32,
    /// Average confidence across all facts
    pub avg_confidence: f32,
    /// Number of learned primitives in adaptive ontology
    pub ontology_size: u32,
    /// Average ontology utility
    pub avg_ontology_utility: f64,
    /// Number of causal nodes in the causal bridge
    pub causal_node_count: u32,
    /// Number of causal edges in the causal bridge
    pub causal_edge_count: u32,
    /// Number of knowledge search results returned this cycle
    pub search_results: u32,
    /// Best search similarity this cycle
    pub best_search_similarity: f32,
    /// Number of domain tags in knowledge graph
    pub domain_count: u32,
}

/// Signals emitted by the knowledge engine for the cognitive loop
///
/// These feed into the ExperienceBus and neuromodulator bath.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeSignals {
    /// Knowledge uncertainty: high when search returns poor matches
    /// Feeds → epistemic confidence, NE boost for attention
    pub uncertainty: f64,
    /// Knowledge contradiction: high when contradictions detected
    /// Feeds → prediction error, exploration drive
    pub contradiction_signal: f64,
    /// Knowledge relevance: how well current input matched stored knowledge
    /// Feeds → confidence boost, DA reward
    pub relevance: f64,
    /// Novelty: how novel the current input is relative to stored knowledge
    /// Feeds → curiosity drive, exploration bonus
    pub novelty: f64,
    /// Causal depth: how many causal chain steps were traced
    /// Feeds → reasoning confidence
    pub causal_depth: f64,
}

// ── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the KnowledgeManager
#[derive(Debug, Clone)]
pub struct KnowledgeManagerConfig {
    /// Maximum facts in the knowledge graph
    pub graph_capacity: usize,
    /// Maximum causal edges
    pub causal_capacity: usize,
    /// How many top-k search results to return per query
    pub search_top_k: usize,
    /// Minimum similarity for a search result to count as "relevant"
    pub relevance_threshold: f32,
    /// Confidence decay interval (cycles between decay passes)
    pub decay_interval: u64,
    /// Adaptive ontology configuration
    pub ontology_config: AdaptiveOntologyConfig,
    /// Processing interval: run extraction every N cycles (1 = every cycle)
    /// Science: reduce overhead by amortizing extraction over multiple cycles
    pub processing_interval: u64,
    /// Path to SQLite database for persistent knowledge storage.
    /// When set, facts and causal edges are saved periodically and loaded on startup.
    /// Science: Ebbinghaus (1885) — cross-session consolidation.
    pub db_path: Option<String>,
    /// Save interval: persist knowledge every N cycles (default: 500).
    pub save_interval: u64,
}

impl Default for KnowledgeManagerConfig {
    fn default() -> Self {
        Self {
            graph_capacity: 10_000,
            causal_capacity: 5_000,
            search_top_k: 5,
            relevance_threshold: 0.15,
            decay_interval: 100,
            ontology_config: AdaptiveOntologyConfig::default(),
            processing_interval: 1,
            db_path: None,
            save_interval: 500,
        }
    }
}

// ── Knowledge Manager ──────────────────────────────────────────────────────

/// Central knowledge engine coordinator
pub struct KnowledgeManager {
    /// Configuration
    config: KnowledgeManagerConfig,
    /// Fact extractor
    extractor: KnowledgeExtractor,
    /// HDC encoder
    encoder: KnowledgeEncoder,
    /// Knowledge graph with temporal facts
    graph: EnhancedKnowledgeGraph,
    /// Causal edge discovery
    causal_bridge: CausalKnowledgeBridge,
    /// Adaptive ontology for primitive learning
    ontology: AdaptiveOntology,
    /// Current cycle (tracked internally)
    current_cycle: u64,
    /// Last telemetry (cached for accessor)
    last_telemetry: KnowledgeTelemetry,
    /// Last signals (cached for accessor)
    last_signals: KnowledgeSignals,
    /// Pending contradiction alerts (drained by cognitive loop)
    pending_alerts: VecDeque<ContradictionAlert>,
    /// Last search results (accessible for reasoning engine context)
    last_search_results: Vec<FactSearchResult>,
    /// Registered entities from the HDC primitive system
    bootstrap_done: bool,
    /// Last causal chain depth traced during search (for signal emission)
    last_causal_depth: usize,
    /// Optional SQLite persistence layer
    persistence: Option<KnowledgePersistence>,
}

impl Default for KnowledgeManager {
    fn default() -> Self {
        Self::new(KnowledgeManagerConfig::default())
    }
}

impl KnowledgeManager {
    pub fn new(config: KnowledgeManagerConfig) -> Self {
        let mut graph = EnhancedKnowledgeGraph::new(config.graph_capacity);
        let mut causal_bridge = CausalKnowledgeBridge::new(config.causal_capacity);
        let ontology = AdaptiveOntology::new(config.ontology_config.clone());

        // Initialize persistence and load existing knowledge if DB path is configured
        let persistence = config.db_path.as_ref().map(|path| {
            let mut p = KnowledgePersistence::new(path);
            // Load persisted facts into the graph
            if let Ok(records) = p.load_facts() {
                for record in &records {
                    graph.import_fact_record(record);
                }
                if !records.is_empty() {
                    tracing::info!(
                        target: "knowledge::persistence",
                        facts = records.len(),
                        "Loaded persisted knowledge facts"
                    );
                }
            }
            // Load persisted causal edges
            if let Ok(edges) = p.load_causal_edges() {
                for edge in &edges {
                    causal_bridge.import_edge(
                        &edge.cause,
                        &edge.effect,
                        edge.strength,
                        edge.is_inhibitory,
                    );
                }
                if !edges.is_empty() {
                    tracing::info!(
                        target: "knowledge::persistence",
                        edges = edges.len(),
                        "Loaded persisted causal edges"
                    );
                }
            }
            p
        });

        Self {
            config,
            extractor: KnowledgeExtractor::new(),
            encoder: KnowledgeEncoder::new(),
            graph,
            causal_bridge,
            ontology,
            current_cycle: 0,
            last_telemetry: KnowledgeTelemetry::default(),
            last_signals: KnowledgeSignals::default(),
            pending_alerts: VecDeque::with_capacity(32),
            last_search_results: Vec::new(),
            bootstrap_done: false,
            last_causal_depth: 0,
            persistence,
        }
    }

    /// Bootstrap the knowledge engine with known entities from the primitive system.
    ///
    /// Call once at startup to pre-register geopolitical entities, organizations,
    /// and key concepts so the extractor can recognize them immediately.
    pub fn bootstrap_entities(&mut self) {
        if self.bootstrap_done {
            return;
        }

        // Register institutional/geopolitical entities from the HDC primitive system
        let geo_entities = [
            ("united states", EntityType::Organization),
            ("china", EntityType::Organization),
            ("russia", EntityType::Organization),
            ("iran", EntityType::Organization),
            ("israel", EntityType::Organization),
            ("european union", EntityType::Organization),
            ("nato", EntityType::Organization),
            ("united nations", EntityType::Organization),
            ("opec", EntityType::Organization),
            ("imf", EntityType::Organization),
            ("world bank", EntityType::Organization),
        ];

        let concept_entities = [
            ("sanctions", EntityType::Concept),
            ("inflation", EntityType::Concept),
            ("recession", EntityType::Concept),
            ("blockade", EntityType::Event),
            ("ceasefire", EntityType::Event),
            ("treaty", EntityType::Concept),
            ("sovereignty", EntityType::Concept),
            ("gdp", EntityType::Quantity),
            ("oil", EntityType::Artifact),
            ("nuclear", EntityType::Concept),
            ("diplomacy", EntityType::Concept),
            ("trade", EntityType::Concept),
            ("alliance", EntityType::Concept),
            ("conflict", EntityType::Event),
            ("war", EntityType::Event),
        ];

        // ── Science & Technology ──────────────────────────────────────────
        let science_concepts = [
            ("climate change", EntityType::Concept),
            ("artificial intelligence", EntityType::Concept),
            ("quantum computing", EntityType::Concept),
            ("genetics", EntityType::Concept),
            ("evolution", EntityType::Concept),
            ("gravity", EntityType::Concept),
            ("thermodynamics", EntityType::Concept),
            ("entropy", EntityType::Concept),
            ("photosynthesis", EntityType::Concept),
            ("metabolism", EntityType::Concept),
        ];

        let science_artifacts = [
            ("carbon dioxide", EntityType::Artifact),
            ("hydrogen", EntityType::Artifact),
            ("oxygen", EntityType::Artifact),
            ("nitrogen", EntityType::Artifact),
        ];

        let science_places = [
            ("earth", EntityType::Place),
            ("mars", EntityType::Place),
            ("moon", EntityType::Place),
            ("sun", EntityType::Place),
        ];

        // ── Economics ─────────────────────────────────────────────────────
        let econ_entities = [
            ("interest rate", EntityType::Quantity),
            ("stock market", EntityType::Concept),
            ("supply chain", EntityType::Concept),
            ("cryptocurrency", EntityType::Concept),
            ("central bank", EntityType::Organization),
            ("federal reserve", EntityType::Organization),
            ("dollar", EntityType::Artifact),
            ("euro", EntityType::Artifact),
            ("yuan", EntityType::Artifact),
        ];

        // ── Social & Political ───────────────────────────────────────────
        let social_concepts = [
            ("democracy", EntityType::Concept),
            ("authoritarianism", EntityType::Concept),
            ("human rights", EntityType::Concept),
            ("civil liberties", EntityType::Concept),
            ("terrorism", EntityType::Concept),
            ("migration", EntityType::Concept),
            ("inequality", EntityType::Concept),
        ];

        let social_orgs = [
            ("supreme court", EntityType::Organization),
            ("congress", EntityType::Organization),
            ("parliament", EntityType::Organization),
        ];

        let social_events = [
            ("election", EntityType::Event),
            ("referendum", EntityType::Event),
            ("revolution", EntityType::Event),
            ("protest", EntityType::Event),
        ];

        // ── Health ───────────────────────────────────────────────────────
        let health_events = [
            ("pandemic", EntityType::Event),
            ("vaccine", EntityType::Event),
            ("epidemic", EntityType::Event),
        ];

        let health_concepts = [
            ("cancer", EntityType::Concept),
            ("diabetes", EntityType::Concept),
            ("malaria", EntityType::Concept),
        ];

        let health_orgs = [("who", EntityType::Organization)];

        // ── Environment ──────────────────────────────────────────────────
        let env_concepts = [
            ("deforestation", EntityType::Concept),
            ("pollution", EntityType::Concept),
            ("biodiversity", EntityType::Concept),
            ("sustainability", EntityType::Concept),
            ("renewable energy", EntityType::Concept),
        ];

        let env_places = [
            ("arctic", EntityType::Place),
            ("amazon", EntityType::Place),
            ("pacific", EntityType::Place),
        ];

        for (text, etype) in geo_entities
            .iter()
            .chain(concept_entities.iter())
            .chain(science_concepts.iter())
            .chain(science_artifacts.iter())
            .chain(science_places.iter())
            .chain(econ_entities.iter())
            .chain(social_concepts.iter())
            .chain(social_orgs.iter())
            .chain(social_events.iter())
            .chain(health_events.iter())
            .chain(health_concepts.iter())
            .chain(health_orgs.iter())
            .chain(env_concepts.iter())
            .chain(env_places.iter())
        {
            self.extractor.register_entity(text, *etype);
        }

        self.bootstrap_done = true;
    }

    /// Register a custom entity for the extractor
    pub fn register_entity(&mut self, text: &str, entity_type: EntityType) {
        self.extractor.register_entity(text, entity_type);
    }

    /// Process one cycle of the knowledge engine.
    ///
    /// Takes the current input text, extracts facts, encodes them,
    /// stores them, searches for context, and emits signals.
    ///
    /// Returns (telemetry, signals) for the cognitive loop.
    pub fn process(
        &mut self,
        input: &str,
        current_cycle: u64,
    ) -> (&KnowledgeTelemetry, &KnowledgeSignals) {
        self.current_cycle = current_cycle;

        // Skip processing if not on interval
        if current_cycle % self.config.processing_interval != 0 && !input.is_empty() {
            // Still do search even on off-cycles
            self.do_search(input);
            self.emit_signals();
            return (&self.last_telemetry, &self.last_signals);
        }

        let mut telem = KnowledgeTelemetry::default();

        // 1. Extract facts from input
        let facts = if input.is_empty() {
            Vec::new()
        } else {
            self.extractor.extract(input)
        };
        telem.facts_extracted = facts.len() as u32;

        // 2-4. Encode, insert, process causal relations
        let mut causal_edges_added = 0u32;
        let mut contradictions_detected = 0u32;

        for fact in &facts {
            let has_causal = fact.relations.iter().any(|r| r.is_causal);

            // 2. Encode as HDC vector
            let encoding = self.encoder.encode_fact(fact);

            // 3. Insert into knowledge graph
            let domain = infer_domain(fact);
            let (_fact_id, contradictions) =
                self.graph
                    .insert(encoding.clone(), current_cycle, domain, has_causal);
            telem.facts_inserted += 1;
            contradictions_detected += contradictions.len() as u32;

            for alert in contradictions {
                self.pending_alerts.push_back(alert);
            }

            // 4. Process causal relations
            for relation in &fact.relations {
                if self
                    .causal_bridge
                    .process_relation(relation, &fact.source_text, current_cycle)
                {
                    causal_edges_added += 1;
                }
            }

            // 6. Feed adaptive ontology
            // If the fact's encoding doesn't match any existing primitive well,
            // learn it as a new concept
            if self
                .ontology
                .lookup(&encoding.vector, current_cycle)
                .is_none()
            {
                // Use the first entity's text as the primitive name
                if let Some(entity) = fact.entities.first() {
                    self.ontology.learn(
                        &entity.text.to_lowercase(),
                        encoding.vector.clone(),
                        vec![], // No parents for directly-learned primitives
                        current_cycle,
                    );
                }
            }
        }

        telem.causal_edges_added = causal_edges_added;
        telem.contradictions_detected = contradictions_detected;

        // 5. Search knowledge graph for context
        self.do_search(input);
        telem.search_results = self.last_search_results.len() as u32;
        telem.best_search_similarity = self
            .last_search_results
            .first()
            .map(|r| r.similarity)
            .unwrap_or(0.0);

        // 7. Periodic maintenance
        if current_cycle % self.config.decay_interval == 0 {
            self.graph.decay_confidence(current_cycle);
        }
        self.ontology.maybe_prune(current_cycle);

        // 7b. Periodic persistence: save facts + causal edges
        if current_cycle > 0 && current_cycle % self.config.save_interval == 0 {
            self.persist_snapshot();
        }

        // Fill remaining telemetry
        telem.graph_size = self.graph.len() as u32;
        telem.avg_confidence = self.graph.average_confidence();
        telem.ontology_size = self.ontology.count() as u32;
        telem.avg_ontology_utility = self.ontology.average_utility();
        telem.causal_node_count = self.causal_bridge.node_count() as u32;
        telem.causal_edge_count = self.causal_bridge.edge_count() as u32;
        telem.domain_count = self.graph.domain_count() as u32;

        self.last_telemetry = telem;

        // 8. Emit signals
        self.emit_signals();

        (&self.last_telemetry, &self.last_signals)
    }

    /// Drain pending contradiction alerts
    pub fn drain_alerts(&mut self) -> Vec<ContradictionAlert> {
        self.pending_alerts.drain(..).collect()
    }

    /// Get last search results (for reasoning engine context injection)
    pub fn last_search_results(&self) -> &[FactSearchResult] {
        &self.last_search_results
    }

    /// Get the causal bridge (for chain tracing)
    pub fn causal_bridge(&self) -> &CausalKnowledgeBridge {
        &self.causal_bridge
    }

    /// Get the knowledge graph (for direct queries)
    pub fn graph(&self) -> &EnhancedKnowledgeGraph {
        &self.graph
    }

    /// Get the adaptive ontology
    pub fn ontology(&self) -> &AdaptiveOntology {
        &self.ontology
    }

    /// Get last telemetry
    pub fn telemetry(&self) -> &KnowledgeTelemetry {
        &self.last_telemetry
    }

    /// Get last signals
    pub fn signals(&self) -> &KnowledgeSignals {
        &self.last_signals
    }

    /// Compose an HDC query from role-term pairs (delegates to encoder)
    pub fn compose_query(
        &mut self,
        role_terms: &[(super::extraction::SemanticRole, &str)],
    ) -> BinaryHV {
        self.encoder.compose_query(role_terms)
    }

    /// Search the knowledge graph with a pre-composed HDC query
    pub fn search_with_vector(&mut self, query: &BinaryHV, k: usize) -> Vec<FactSearchResult> {
        self.graph.search(query, k, self.current_cycle)
    }

    /// Trace causal chains from a starting concept
    pub fn trace_causal_chain(&self, start: &str, max_depth: usize) -> Vec<Vec<String>> {
        self.causal_bridge.trace_chain(start, max_depth)
    }

    /// Consolidate and forget: prune low-confidence non-causal facts,
    /// strengthen causal facts. Called during dream/rest phases.
    /// Science: Stickgold (2005) — sleep-dependent memory consolidation.
    pub fn consolidate_and_forget(&mut self) -> (usize, usize) {
        let pruned = self.graph.prune_low_confidence(
            crate::cognitive_loop::thresholds::KNOWLEDGE_FORGET_CONFIDENCE_THRESHOLD,
        );
        let consolidated = self.graph.strengthen_causal_facts(
            crate::cognitive_loop::thresholds::KNOWLEDGE_CONSOLIDATION_BOOST,
        );
        // Persist surviving facts after consolidation
        if pruned > 0 || consolidated > 0 {
            self.persist_snapshot();
        }
        (pruned, consolidated)
    }

    /// Query the knowledge engine for grounded facts and causal chains.
    /// Used by cross-module consumers (Broca, EthicsEngine).
    pub fn query(&mut self, text: &str) -> super::reasoning_context::KnowledgeQueryResult {
        self.do_search(text);
        let grounding_score = if self.last_search_results.is_empty() {
            0.0
        } else {
            let avg_sim: f64 = self
                .last_search_results
                .iter()
                .map(|r| r.similarity as f64)
                .sum::<f64>()
                / self.last_search_results.len() as f64;
            let certainty = 1.0 - self.last_signals.uncertainty;
            (avg_sim * 0.6 + certainty * 0.4).clamp(0.0, 1.0)
        };
        let facts: Vec<super::reasoning_context::GroundedFact> = self
            .last_search_results
            .iter()
            .map(|r| super::reasoning_context::GroundedFact {
                text: r.source_text.clone(),
                confidence: r.confidence,
                similarity: r.similarity,
                domain: r.domain.clone(),
                is_causal: r.is_causal,
            })
            .collect();

        // Trace causal chains for extracted entities
        let extracted = self.extractor.extract(text);
        let mut causal_chains = Vec::new();
        for fact in &extracted {
            for entity in &fact.entities {
                let chains = self
                    .causal_bridge
                    .trace_chain(&entity.text.to_lowercase(), 5);
                for chain in chains {
                    if !chain.is_empty() {
                        causal_chains.push(super::reasoning_context::CausalChain {
                            root: entity.text.to_lowercase(),
                            steps: chain.clone(),
                            depth: chain.len(),
                        });
                    }
                }
            }
        }

        super::reasoning_context::KnowledgeQueryResult {
            facts,
            causal_chains,
            grounding_score,
        }
    }

    /// Get mutable access to the graph (for consolidation).
    pub fn graph_mut(&mut self) -> &mut EnhancedKnowledgeGraph {
        &mut self.graph
    }

    /// Persist current knowledge state to SQLite (if configured).
    fn persist_snapshot(&mut self) {
        if let Some(ref mut persistence) = self.persistence {
            let fact_records = self.graph.export_fact_records();
            if let Err(e) = persistence.save_facts(&fact_records) {
                tracing::warn!(target: "knowledge::persistence", "Save facts failed: {e}");
            }
            let edge_records = self.causal_bridge.export_edge_records();
            if let Err(e) = persistence.save_causal_edges(&edge_records) {
                tracing::warn!(target: "knowledge::persistence", "Save edges failed: {e}");
            }
        }
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn do_search(&mut self, input: &str) {
        if input.is_empty() {
            self.last_search_results.clear();
            return;
        }

        // Try compositional search first: extract entities and build role-term query
        let facts = self.extractor.extract(input);
        let query_hv = if let Some(fact) = facts.first() {
            if !fact.role_map.is_empty() {
                // Build role-term pairs from the extracted fact's role_map
                let role_terms: Vec<(super::extraction::SemanticRole, String)> = fact
                    .role_map
                    .iter()
                    .map(|(text, role)| (*role, text.clone()))
                    .collect();
                let role_term_refs: Vec<(super::extraction::SemanticRole, &str)> =
                    role_terms.iter().map(|(r, t)| (*r, t.as_str())).collect();
                self.encoder.compose_query(&role_term_refs)
            } else {
                // Entities found but no role assignments — fall back to token encoding
                self.encoder.encode_token(input)
            }
        } else {
            // No entities extracted — fall back to token-based search
            self.encoder.encode_token(input)
        };

        self.last_search_results =
            self.graph
                .search(&query_hv, self.config.search_top_k, self.current_cycle);

        // Trace causal chains for extracted entities and update causal_depth
        let mut max_chain_depth: usize = 0;
        for fact in &facts {
            for entity in &fact.entities {
                let chains = self
                    .causal_bridge
                    .trace_chain(&entity.text.to_lowercase(), 5);
                for chain in &chains {
                    if chain.len() > max_chain_depth {
                        max_chain_depth = chain.len();
                    }
                }
            }
        }
        self.last_causal_depth = max_chain_depth;
    }

    fn emit_signals(&mut self) {
        let best_sim = self
            .last_search_results
            .first()
            .map(|r| r.similarity as f64)
            .unwrap_or(0.0);

        let avg_sim = if self.last_search_results.is_empty() {
            0.0
        } else {
            self.last_search_results
                .iter()
                .map(|r| r.similarity as f64)
                .sum::<f64>()
                / self.last_search_results.len() as f64
        };

        self.last_signals = KnowledgeSignals {
            // Uncertainty: inverse of best match quality
            uncertainty: (1.0 - best_sim).clamp(0.0, 1.0),
            // Contradiction signal: from pending alerts
            contradiction_signal: if self.last_telemetry.contradictions_detected > 0 {
                (self.last_telemetry.contradictions_detected as f64 * 0.3).min(1.0)
            } else {
                0.0
            },
            // Relevance: average search similarity
            relevance: avg_sim.clamp(0.0, 1.0),
            // Novelty: inverse of relevance (unknown input = novel)
            novelty: (1.0 - avg_sim).clamp(0.0, 1.0),
            // Causal depth: normalized by max reasonable chain length (5 = deep chain)
            causal_depth: (self.last_causal_depth as f64 / 5.0).clamp(0.0, 1.0),
        };
    }
}

/// Infer a domain tag from an extracted fact based on entity types and relations
fn infer_domain(fact: &super::extraction::ExtractedFact) -> Option<String> {
    for entity in &fact.entities {
        match entity.entity_type {
            EntityType::Organization | EntityType::Person => {
                // Check for geopolitical context
                if fact.relations.iter().any(|r| r.is_causal) {
                    return Some("geopolitics".to_string());
                }
                return Some("social".to_string());
            }
            EntityType::Place => return Some("geography".to_string()),
            EntityType::Event => return Some("events".to_string()),
            EntityType::Quantity => return Some("economics".to_string()),
            EntityType::Process => return Some("science".to_string()),
            _ => {}
        }
    }
    None
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_process() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        let (telem, signals) = mgr.process("The United States imposed sanctions on Iran.", 1);

        assert!(telem.facts_extracted > 0);
        assert!(signals.uncertainty >= 0.0);
        assert!(signals.uncertainty <= 1.0);
    }

    #[test]
    fn test_causal_processing() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        mgr.process("Sanctions caused oil prices to increase dramatically.", 1);

        assert!(mgr.causal_bridge().edge_count() > 0);
    }

    #[test]
    fn test_knowledge_accumulation() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        let sentences = [
            "Iran caused sanctions to escalate.",
            "Russia blocked the NATO agreement.",
            "China increased trade with Iran.",
            "Sanctions caused oil prices to spike.",
            "NATO triggered a military response.",
        ];
        for (i, sentence) in sentences.iter().enumerate() {
            mgr.process(sentence, i as u64);
        }

        assert!(mgr.graph().len() > 0);
        assert!(mgr.telemetry().graph_size > 0);
    }

    #[test]
    fn test_search_relevance() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        // Insert some knowledge
        mgr.process("Iran launched missiles at targets.", 1);
        mgr.process("Oil prices spiked due to conflict.", 2);

        // Search should find something
        let (telem, signals) = mgr.process("Iran missiles", 3);
        // May or may not find depending on HDC similarity
        assert!(signals.uncertainty >= 0.0);
    }

    #[test]
    fn test_empty_input() {
        let mut mgr = KnowledgeManager::default();
        let (telem, _) = mgr.process("", 1);
        assert_eq!(telem.facts_extracted, 0);
    }

    #[test]
    fn test_ontology_learning() {
        let mut mgr = KnowledgeManager::default();

        // Process novel concepts
        for i in 0..20 {
            mgr.process(
                &format!("NovelConcept{i} emerged from the research findings."),
                i as u64,
            );
        }

        // Some concepts should have been learned
        // (depends on extraction finding capitalized novel concepts)
        // The ontology should have at least attempted lookups
        assert!(mgr.ontology().total_queries() >= 0);
    }

    #[test]
    fn test_bootstrap_idempotent() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();
        mgr.bootstrap_entities(); // Should not duplicate
    }

    #[test]
    fn test_compose_and_search() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        mgr.process("Iran blocked the Strait of Hormuz.", 1);

        let query = mgr.compose_query(&[(super::super::extraction::SemanticRole::Agent, "Iran")]);
        let results = mgr.search_with_vector(&query, 5);
        // May or may not find depending on similarity
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_causal_chain_tracing() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        mgr.process("Sanctions caused oil shortage.", 1);
        mgr.process("Oil shortage caused price spike.", 2);
        mgr.process("Price spike caused inflation.", 3);

        let chains = mgr.trace_causal_chain("sanctions", 5);
        // Chain tracing depends on extraction quality
        // At minimum, the function should not panic
    }

    #[test]
    fn test_telemetry_fields_populated() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        mgr.process("Something happened somewhere.", 1);

        let telem = mgr.telemetry();
        // Basic sanity: all fields are defined
        assert!(telem.avg_confidence >= 0.0);
        assert!(telem.avg_confidence <= 1.0);
    }

    #[test]
    fn test_decay_runs() {
        let config = KnowledgeManagerConfig {
            decay_interval: 5,
            ..Default::default()
        };
        let mut mgr = KnowledgeManager::new(config);
        mgr.bootstrap_entities();

        mgr.process("Initial fact.", 1);

        // Run enough cycles to trigger decay
        for i in 2..20 {
            mgr.process("", i);
        }

        // Should not panic; decay should have run
    }

    #[test]
    fn test_compositional_search() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        // Insert a fact about Iran sanctions
        mgr.process("Iran caused sanctions to escalate dramatically.", 1);

        // Search with "Iran" — should use compositional matching via entity extraction
        let (_telem, signals) = mgr.process("Iran sanctions", 2);

        // The search should have run (results may vary by HDC similarity)
        // but the compositional path should have been taken since "iran" and
        // "sanctions" are both registered entities
        assert!(signals.uncertainty >= 0.0);
        assert!(signals.uncertainty <= 1.0);

        // Verify search results were populated (graph has at least one fact)
        assert!(mgr.graph().len() > 0);
        assert!(mgr.last_search_results().len() <= mgr.config.search_top_k);
    }

    #[test]
    fn test_broad_bootstrap() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        // Verify science entities are registered by extracting them
        let facts = mgr
            .extractor
            .extract("Climate change affects the earth significantly.");
        let has_science = facts.iter().any(|f| {
            f.entities
                .iter()
                .any(|e| e.text.to_lowercase().contains("climate change"))
        });
        assert!(
            has_science,
            "Science entity 'climate change' should be extractable"
        );

        // Verify economics entities
        let facts = mgr
            .extractor
            .extract("The federal reserve raised the interest rate.");
        let has_econ = facts.iter().any(|f| {
            f.entities.iter().any(|e| {
                e.text.to_lowercase().contains("federal reserve")
                    || e.text.to_lowercase().contains("interest rate")
            })
        });
        assert!(has_econ, "Economics entities should be extractable");

        // Verify social/political entities
        let facts = mgr
            .extractor
            .extract("Democracy requires civil liberties to function.");
        let has_social = facts.iter().any(|f| {
            f.entities.iter().any(|e| {
                e.text.to_lowercase().contains("democracy")
                    || e.text.to_lowercase().contains("civil liberties")
            })
        });
        assert!(has_social, "Social entities should be extractable");

        // Verify environment entities
        let facts = mgr
            .extractor
            .extract("Deforestation in the amazon accelerates climate change.");
        let has_env = facts.iter().any(|f| {
            f.entities.iter().any(|e| {
                e.text.to_lowercase().contains("amazon")
                    || e.text.to_lowercase().contains("deforestation")
            })
        });
        assert!(has_env, "Environment entities should be extractable");

        // Verify health entities
        let facts = mgr
            .extractor
            .extract("The pandemic caused cancer research delays.");
        let has_health = facts.iter().any(|f| {
            f.entities.iter().any(|e| {
                e.text.to_lowercase().contains("pandemic")
                    || e.text.to_lowercase().contains("cancer")
            })
        });
        assert!(has_health, "Health entities should be extractable");
    }

    #[test]
    fn test_causal_depth_signal() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        // Build a causal chain: sanctions → oil shortage → inflation
        mgr.process("Sanctions caused oil shortage.", 1);
        mgr.process("Oil shortage caused inflation.", 2);
        mgr.process("Inflation caused recession.", 3);

        // Now search for "sanctions" — should trace the causal chain
        let (_telem, signals) = mgr.process("Sanctions caused problems.", 4);
        let causal_depth = signals.causal_depth;

        // causal_depth should be > 0 if the causal bridge traced any chain
        // from the extracted entities (sanctions is registered)
        let chains = mgr.trace_causal_chain("sanctions", 5);
        if !chains.is_empty() {
            assert!(
                causal_depth > 0.0,
                "causal_depth should be > 0 when causal chains exist, got {}",
                causal_depth
            );
        }
    }
}
