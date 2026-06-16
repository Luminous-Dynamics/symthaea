// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
use super::persistence::{CausalEdgeRecord, KnowledgePersistence, OntologyRecord};
use super::reasoning_context::{KnowledgeQueryResult, ReasoningContext};
use std::collections::VecDeque;
use symthaea_core::hdc::unified_hv::BinaryHV;

// ── Calibration Audit ──────────────────────────────────────────────────────

/// Calibration audit: tracks whether knowledge confidence scores are well-calibrated.
/// Maintains a 10-bin histogram of (confidence_bucket, count, correct_count).
/// Science: Guo et al. (2017) — On Calibration of Modern Neural Networks.
#[derive(Debug, Clone)]
pub struct CalibrationAudit {
    /// 10 bins: [0.0-0.1), [0.1-0.2), ..., [0.9-1.0]
    /// Each bin: (total_predictions, correct_predictions)
    bins: [(u64, u64); 10],
    /// Total samples tracked
    total_samples: u64,
}

impl Default for CalibrationAudit {
    fn default() -> Self {
        Self {
            bins: [(0, 0); 10],
            total_samples: 0,
        }
    }
}

impl CalibrationAudit {
    /// Record a prediction outcome: the fact had this confidence, and the prediction was correct/incorrect.
    pub fn record(&mut self, confidence: f32, correct: bool) {
        let bin = (confidence * 10.0).floor().min(9.0) as usize;
        self.bins[bin].0 += 1;
        if correct {
            self.bins[bin].1 += 1;
        }
        self.total_samples += 1;
    }

    /// Expected Calibration Error: weighted average of |accuracy - confidence| per bin.
    /// Lower is better (0.0 = perfectly calibrated).
    pub fn ece(&self) -> f64 {
        if self.total_samples == 0 {
            return 0.0;
        }

        let mut ece = 0.0;
        for (i, &(total, correct)) in self.bins.iter().enumerate() {
            if total == 0 {
                continue;
            }
            let bin_confidence = (i as f64 + 0.5) / 10.0; // Bin midpoint
            let bin_accuracy = correct as f64 / total as f64;
            debug_assert!(
                self.total_samples > 0,
                "total_samples checked at function entry"
            );
            let weight = total as f64 / self.total_samples as f64;
            ece += weight * (bin_accuracy - bin_confidence).abs();
        }
        ece
    }

    /// Maximum Calibration Error: worst bin's |accuracy - confidence|.
    pub fn mce(&self) -> f64 {
        self.bins
            .iter()
            .enumerate()
            .filter(|&(_, &(total, _))| total > 0)
            .map(|(i, &(total, correct))| {
                let bin_confidence = (i as f64 + 0.5) / 10.0;
                let bin_accuracy = correct as f64 / total as f64;
                (bin_accuracy - bin_confidence).abs()
            })
            .fold(0.0_f64, f64::max)
    }

    /// Per-bin summary: (bin_midpoint, total_count, accuracy).
    pub fn bin_summary(&self) -> Vec<(f32, u64, f32)> {
        self.bins
            .iter()
            .enumerate()
            .filter(|&(_, &(total, _))| total > 0)
            .map(|(i, &(total, correct))| {
                let midpoint = (i as f32 + 0.5) / 10.0;
                let accuracy = if total > 0 {
                    correct as f32 / total as f32
                } else {
                    0.0
                };
                (midpoint, total, accuracy)
            })
            .collect()
    }

    /// Total samples tracked.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }
}

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
    /// Expected Calibration Error (0.0 = perfect, 1.0 = worst)
    pub calibration_ece: f64,
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
    /// Epistemic surprise: high when contradictions + novelty are both present.
    /// Feeds → FEP learning signal boost (Friston 2010).
    pub epistemic_surprise: f64,
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
    /// Optional path for SQLite-backed knowledge persistence.
    /// When `None`, knowledge is in-memory only (lost on restart).
    pub db_path: Option<String>,
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
    /// Save interval from config (cycles between persistence snapshots)
    save_interval: u64,
    /// Ontology learning rate multiplier, modulated by prediction error
    /// Science: Rescorla-Wagner (1972) — PE modulates learning rate
    ontology_lr_multiplier: f32,
    /// Calibration audit: tracks ECE across prediction outcomes
    calibration_audit: CalibrationAudit,
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
        let mut ontology = AdaptiveOntology::new(config.ontology_config.clone());

        // Initialize persistence and load existing knowledge
        let persistence = config.db_path.as_ref().map(|path| {
            let mut p = KnowledgePersistence::new(path);
            // Load existing facts
            if let Ok(facts) = p.load_facts() {
                for record in &facts {
                    graph.import_fact_record(record);
                }
                if !facts.is_empty() {
                    tracing::info!(count = facts.len(), "Knowledge: loaded facts from SQLite");
                }
            }
            // Load existing causal edges
            if let Ok(edges) = p.load_causal_edges() {
                let edge_count = edges.len();
                for record in &edges {
                    causal_bridge.import_edge(
                        record.cause.clone(),
                        record.effect.clone(),
                        record.strength,
                    );
                }
                if edge_count > 0 {
                    tracing::info!(
                        count = edge_count,
                        "Knowledge: loaded causal edges from SQLite"
                    );
                }
            }
            // Load existing ontology primitives
            if let Ok(records) = p.load_ontology() {
                let onto_count = records.len();
                for record in &records {
                    ontology.import_ontology_record(record);
                }
                if onto_count > 0 {
                    tracing::info!(
                        count = onto_count,
                        "Knowledge: loaded ontology primitives from SQLite"
                    );
                }
            }
            p
        });

        let save_interval = crate::cognitive_loop::thresholds::KNOWLEDGE_SAVE_INTERVAL;

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
            save_interval,
            ontology_lr_multiplier: 1.0,
            calibration_audit: CalibrationAudit::default(),
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

            // Auto-detect IS-A relations from extracted text.
            // Patterns: "X is a Y", "X is a type of Y", "X are Y"
            // Science: Quillian (1967) — semantic networks.
            for relation in &fact.relations {
                let pred = relation.predicate.to_lowercase();
                let is_isa = pred == "is"
                    || pred == "is a"
                    || pred == "is an"
                    || pred == "are"
                    || pred.contains("type of")
                    || pred.contains("kind of")
                    || pred.contains("form of")
                    || pred.contains("subclass of");

                if is_isa && !relation.is_negated && relation.confidence > 0.3 {
                    let child = relation.subject.to_lowercase();
                    let parent = relation.object.to_lowercase();
                    // First ensure both exist in ontology
                    if self
                        .ontology
                        .lookup(&self.encoder.encode_token(&child), current_cycle)
                        .is_some()
                    {
                        self.ontology.set_is_a(&child, &parent);
                    } else {
                        // Learn child, then set IS-A
                        let child_hv = self.encoder.encode_token(&child);
                        self.ontology
                            .learn(&child, child_hv, vec![parent.clone()], current_cycle);
                        self.ontology.set_is_a(&child, &parent);
                    }
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

        // AGM-style contradiction resolution: demote weaker fact confidence
        // (Alchourrón, Gärdenfors & Makinson 1985)
        if !self.pending_alerts.is_empty() {
            let threshold =
                crate::cognitive_loop::thresholds::KNOWLEDGE_CONTRADICTION_RESOLUTION_THRESHOLD;
            let resolvable: Vec<_> = self
                .pending_alerts
                .iter()
                .filter(|a| a.similarity > threshold)
                .cloned()
                .collect();
            if !resolvable.is_empty() {
                self.graph.resolve_contradictions(&resolvable);
            }
        }

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

        // Fill remaining telemetry
        telem.graph_size = self.graph.len() as u32;
        telem.avg_confidence = self.graph.average_confidence();
        telem.ontology_size = self.ontology.count() as u32;
        telem.avg_ontology_utility = self.ontology.average_utility();
        telem.causal_node_count = self.causal_bridge.node_count() as u32;
        telem.causal_edge_count = self.causal_bridge.edge_count() as u32;
        telem.domain_count = self.graph.domain_count() as u32;
        telem.calibration_ece = self.calibration_audit.ece();

        self.last_telemetry = telem;

        // 8. Emit signals
        self.emit_signals();

        // 9. Periodic persistence snapshot
        if current_cycle > 0 && current_cycle % self.save_interval == 0 {
            self.persist_snapshot();
        }

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

    // ── Persistence & Consolidation ─────────────────────────────────────

    /// Persist current knowledge state to SQLite.
    ///
    /// Saves all facts and causal edges. No-op if persistence is not configured.
    pub fn persist_snapshot(&mut self) {
        if let Some(ref mut p) = self.persistence {
            let facts = self.graph.export_fact_records();
            let edge_tuples = self.causal_bridge.export_edge_records();
            let edges: Vec<CausalEdgeRecord> = edge_tuples
                .into_iter()
                .map(|(cause, effect, strength)| CausalEdgeRecord {
                    cause,
                    effect,
                    strength,
                    is_inhibitory: false,
                    cycle: 0,
                })
                .collect();
            let ontology_records = self.ontology.export_ontology_records();
            if let Err(e) = p.save_facts(&facts) {
                tracing::warn!(error = %e, "Knowledge persistence: failed to save facts");
            }
            if let Err(e) = p.save_causal_edges(&edges) {
                tracing::warn!(error = %e, "Knowledge persistence: failed to save edges");
            }
            if let Err(e) = p.save_ontology(&ontology_records) {
                tracing::warn!(error = %e, "Knowledge persistence: failed to save ontology");
            }
        }
    }

    /// Apply dream consolidation feedback to the knowledge graph.
    ///
    /// When the dream engine completes a consolidation cycle with sufficient quality,
    /// this method strengthens knowledge graph edges whose source text contains any
    /// of the consolidated topic keywords.  The boost magnitude is
    /// `DREAM_KNOWLEDGE_REPLAY_WEIGHT * quality` — proportional to how effective the
    /// dream replay was, matching Rasch & Born (2013) targeted-reactivation findings.
    ///
    /// # Arguments
    /// - `consolidated_topics`: keywords inferred from recently consolidated memories.
    ///   If empty, falls back to strengthening all causal facts (non-targeted replay).
    /// - `quality`: 0.0–1.0 score representing dream consolidation effectiveness
    ///   (typically `best_phi_improvement` from `DreamResult`).
    ///
    /// Returns the number of facts strengthened.
    ///
    /// Science:
    /// - Rasch & Born (2013) — targeted reactivation during NREM strengthens cued traces.
    /// - Tononi & Cirelli (2006) — synaptic homeostasis: only effective consolidation
    ///   modifies weights.
    pub fn apply_dream_consolidation(
        &mut self,
        consolidated_topics: &[String],
        quality: f64,
    ) -> usize {
        let min_quality = crate::cognitive_loop::thresholds::DREAM_KNOWLEDGE_MIN_QUALITY;
        if quality < min_quality {
            return 0;
        }
        let replay_weight = crate::cognitive_loop::thresholds::DREAM_KNOWLEDGE_REPLAY_WEIGHT;
        let boost = (replay_weight * quality) as f32;

        let strengthened = if consolidated_topics.is_empty() {
            // Non-targeted replay: fall back to strengthening all causal facts
            self.graph.strengthen_causal_facts(boost)
        } else {
            // Targeted replay: boost only facts relevant to consolidated topics
            self.graph
                .strengthen_facts_by_topic(consolidated_topics, boost)
        };

        if strengthened > 0 {
            tracing::debug!(
                strengthened,
                quality,
                boost,
                topics = consolidated_topics.len(),
                "Dream→Knowledge: strengthened facts after dream consolidation"
            );
        }
        strengthened
    }

    /// Consolidate knowledge during dream/rest: prune weak facts, strengthen causal ones.
    ///
    /// Science: Stickgold & Walker (2013) — sleep-dependent memory consolidation
    /// Returns (pruned_count, strengthened_count).
    pub fn consolidate_and_forget(&mut self) -> (usize, usize) {
        let pruned = self.graph.prune_low_confidence(
            crate::cognitive_loop::thresholds::KNOWLEDGE_FORGET_CONFIDENCE_THRESHOLD,
        );
        let consolidated = self.graph.strengthen_causal_facts(
            crate::cognitive_loop::thresholds::KNOWLEDGE_CONSOLIDATION_BOOST,
        );
        if pruned > 0 || consolidated > 0 {
            self.persist_snapshot();
        }
        (pruned, consolidated)
    }

    /// Query the knowledge engine for cross-module consumers (Broca, Ethics).
    ///
    /// Returns a lightweight result with grounded facts and causal chains.
    pub fn query(&self, input: &str) -> KnowledgeQueryResult {
        let facts: Vec<super::reasoning_context::GroundedFact> = self
            .last_search_results
            .iter()
            .map(|r| super::reasoning_context::GroundedFact {
                text: format!("fact:{}", r.fact_id),
                confidence: r.confidence,
                similarity: r.similarity,
                domain: None,
                is_causal: false,
            })
            .collect();

        let causal_chains: Vec<super::reasoning_context::CausalChain> = input
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .flat_map(|term| {
                let lower = term.to_lowercase();
                self.causal_bridge
                    .trace_chain(&lower, 5)
                    .into_iter()
                    .filter(|c| !c.is_empty())
                    .map(move |steps| super::reasoning_context::CausalChain {
                        root: lower.clone(),
                        steps: steps.clone(),
                        depth: steps.len(),
                    })
            })
            .collect();

        let grounding_score = if facts.is_empty() {
            0.0
        } else {
            facts.iter().map(|f| f.similarity as f64).sum::<f64>() / facts.len() as f64
        };

        KnowledgeQueryResult {
            facts,
            causal_chains,
            grounding_score,
        }
    }

    /// Get mutable access to the knowledge graph (for persistence import).
    pub fn graph_mut(&mut self) -> &mut EnhancedKnowledgeGraph {
        &mut self.graph
    }

    /// Get the source texts of the top-k highest-confidence facts.
    ///
    /// Used by episodic memory bridge during dream consolidation to promote
    /// high-confidence knowledge into episodic memory for long-term retention.
    pub fn top_facts(&self, k: usize) -> Vec<String> {
        let mut facts: Vec<_> = self.graph.all_facts().collect();
        facts.sort_by(|a, b| {
            b.encoding
                .confidence
                .partial_cmp(&a.encoding.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        facts
            .iter()
            .take(k)
            .map(|f| f.encoding.source_text.clone())
            .collect()
    }

    /// Top-k facts filtered by relevance to last search query.
    /// Combines confidence (60%) and similarity (40%) for RAG-style grounding.
    /// Science: Lewis et al. (2020) — retrieval-augmented generation.
    pub fn top_grounded_facts(&self, k: usize) -> Vec<String> {
        if self.last_search_results.is_empty() {
            return self.top_facts(k);
        }

        let mut scored: Vec<(u64, f32)> = self
            .last_search_results
            .iter()
            .map(|r| {
                let combined = r.confidence * 0.6 + r.similarity * 0.4;
                (r.fact_id, combined)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .iter()
            .take(k)
            .filter_map(|(id, _)| {
                self.graph
                    .all_facts()
                    .find(|f| f.id == *id)
                    .map(|f| f.encoding.source_text.clone())
            })
            .collect()
    }

    /// Search for facts within a cycle window (temporal scoping).
    /// Returns facts inserted within [start_cycle, end_cycle].
    /// Science: Tulving (1972) — episodic vs semantic temporal context.
    pub fn search_temporal_window(
        &self,
        start_cycle: u64,
        end_cycle: u64,
        k: usize,
    ) -> Vec<FactSearchResult> {
        self.graph
            .all_facts()
            .filter(|f| f.inserted_at_cycle >= start_cycle && f.inserted_at_cycle <= end_cycle)
            .map(|f| FactSearchResult {
                fact_id: f.id,
                similarity: 1.0, // No query vector — temporal filter only
                confidence: f.confidence,
            })
            .take(k)
            .collect()
    }

    /// Search combining HDC similarity AND temporal recency.
    /// Recent facts get a recency boost: score = similarity * (0.7 + 0.3 * recency).
    /// Science: Anderson & Schooler (1991) — rational analysis of memory.
    pub fn search_with_recency(
        &mut self,
        query: &BinaryHV,
        k: usize,
        recency_window: u64,
    ) -> Vec<FactSearchResult> {
        let mut results = self.graph.search(query, k * 2, self.current_cycle);

        // Apply recency boost
        for r in &mut results {
            if let Some(fact) = self.graph.all_facts().find(|f| f.id == r.fact_id) {
                let age = self.current_cycle.saturating_sub(fact.last_accessed_cycle);
                let recency = if recency_window > 0 {
                    1.0 - (age as f32 / recency_window as f32).min(1.0)
                } else {
                    0.0
                };
                r.similarity *= 0.7 + 0.3 * recency;
            }
        }

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }

    /// Calibrate fact confidence based on prediction outcomes.
    /// Accurate predictions boost confidence, inaccurate reduce.
    /// Science: Jaynes (2003) — probability as extended logic.
    pub fn calibrate_from_prediction(&mut self, prediction_error: f32) {
        let boost = if prediction_error < 0.3 {
            0.01 // Good prediction → slightly boost confidence
        } else if prediction_error > 0.7 {
            -0.02 // Bad prediction → reduce confidence
        } else {
            0.0
        };

        if boost != 0.0 {
            for result in &self.last_search_results {
                self.graph.adjust_confidence(result.fact_id, boost);
            }
        }

        // Record calibration data for each search result
        let correct = prediction_error < 0.5;
        for result in &self.last_search_results {
            self.calibration_audit.record(result.confidence, correct);
        }

        // Causal strength learning from prediction outcomes.
        // Science: Rescorla-Wagner (1972) — prediction error drives causal learning.
        let outcome_good = prediction_error < 0.4;
        let causal_lr = 0.05 * (1.0 - prediction_error).max(0.1);

        // Phase 1: collect causal-eligible fact first words from graph (immutable borrow).
        // Gather fact_ids first to avoid holding &self.last_search_results across the graph borrow.
        let search_fact_ids: Vec<u64> =
            self.last_search_results.iter().map(|r| r.fact_id).collect();
        let causal_fact_words: Vec<String> = search_fact_ids
            .iter()
            .filter_map(|&fact_id| {
                self.graph
                    .all_facts()
                    .find(|f| f.id == fact_id)
                    .filter(|f| f.has_causal_relations)
                    .map(|f| {
                        f.encoding
                            .source_text
                            .split_whitespace()
                            .next()
                            .unwrap_or("")
                            .to_string()
                    })
            })
            .filter(|w| !w.is_empty())
            .collect();

        // Phase 2: look up edges and collect (cause, effect) pairs (immutable borrow of bridge).
        let causal_updates: Vec<(String, String)> = causal_fact_words
            .iter()
            .filter_map(|first_word| {
                let effects = self.causal_bridge.effects_of(first_word);
                effects.first().map(|e| (e.cause.clone(), e.effect.clone()))
            })
            .collect();

        // Phase 3: apply updates (mutable borrow of bridge only).
        for (cause, effect) in &causal_updates {
            self.causal_bridge
                .update_strength_from_outcome(cause, effect, outcome_good, causal_lr);
        }
    }

    /// Get the calibration audit for external monitoring.
    pub fn calibration_audit(&self) -> &CalibrationAudit {
        &self.calibration_audit
    }

    /// Set ontology learning rate from prediction error.
    /// High PE → faster learning; low PE → slower (already learned).
    /// Science: Rescorla-Wagner (1972).
    pub fn set_ontology_lr_from_pe(&mut self, prediction_error: f32) {
        self.ontology_lr_multiplier = (0.5 + prediction_error * 1.5).clamp(0.3, 2.0);
    }

    /// Modulate knowledge learning rate by consciousness level.
    /// High Psi → boosted plasticity (conscious moments are more memorable).
    /// Science: Dehaene et al. (2006) — conscious access enhances learning and memory.
    pub fn modulate_lr_from_consciousness(&mut self, unified_psi: f64) {
        let consciousness_factor = (0.85 + unified_psi * 0.3).clamp(0.85, 1.15) as f32;
        self.ontology_lr_multiplier *= consciousness_factor;
        self.ontology_lr_multiplier = self.ontology_lr_multiplier.clamp(0.3, 2.5);
    }

    /// Get per-domain uncertainty: (domain_name, avg_confidence, fact_count).
    /// Sorted by confidence ascending (most uncertain first).
    pub fn domain_uncertainty(&self) -> Vec<(String, f32, usize)> {
        self.graph.domain_distribution()
    }

    /// Counterfactual query: find causes for an effect with necessity scores.
    /// Science: Pearl (2000) — do-calculus.
    pub fn counterfactual(&self, effect: &str, max_results: usize) -> Vec<(String, f32)> {
        self.causal_bridge
            .counterfactual_necessity(effect, max_results)
    }

    /// Simulate an intervention: do(X) → propagated effects.
    /// Science: Pearl (2009) — do-calculus.
    pub fn do_intervention(&self, intervention: &str, max_depth: usize) -> Vec<(String, f32)> {
        self.causal_bridge.do_intervention(intervention, max_depth)
    }

    /// Detect potential confounders between two entities.
    /// Science: Pearl (2000) — backdoor criterion.
    pub fn detect_confounders(&self, entity_a: &str, entity_b: &str) -> Vec<(String, f32)> {
        self.causal_bridge.detect_confounders(entity_a, entity_b)
    }

    /// Update causal edge strength based on prediction outcome.
    pub fn update_causal_strength(
        &mut self,
        cause: &str,
        effect: &str,
        outcome_occurred: bool,
        learning_rate: f32,
    ) {
        self.causal_bridge.update_strength_from_outcome(
            cause,
            effect,
            outcome_occurred,
            learning_rate,
        );
    }

    /// Find analogous facts in a target domain using HDC similarity.
    /// Science: Gentner (1983) — structure mapping theory.
    pub fn find_analogous(
        &self,
        source_id: u64,
        target_domain: &str,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        // Find the source fact's vector (clone to release borrow)
        let source_vector = match self.graph.all_facts().find(|f| f.id == source_id) {
            Some(f) => f.encoding.vector.clone(),
            None => return Vec::new(),
        };

        // Search for similar facts in the target domain
        let results = self
            .graph
            .search_domain(target_domain, &source_vector, top_k * 3);
        results
            .iter()
            .take(top_k)
            .map(|r| (format!("fact:{}", r.fact_id), r.similarity))
            .collect()
    }

    /// Get per-domain distribution: (domain_name, fact_count, avg_confidence).
    pub fn domain_distribution(&self) -> Vec<(String, usize, f32)> {
        self.graph
            .domain_distribution()
            .into_iter()
            .map(|(name, avg_conf, count)| (name, count, avg_conf))
            .collect()
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

        // Spreading activation: expand search through HDC similarity neighbors.
        // Science: Anderson (1983) ACT-R spreading activation.
        if !self.last_search_results.is_empty() && self.graph.all_facts().count() > 5 {
            let seeds = self.last_search_results.clone();
            let spread_results = self.graph.spreading_activation_search(
                &seeds,
                2,   // 2 hops
                0.5, // 50% decay per hop
                self.config.search_top_k * 2,
                self.current_cycle,
            );

            // Merge: keep original results but add novel spread activations
            for spread in &spread_results {
                if !self
                    .last_search_results
                    .iter()
                    .any(|r| r.fact_id == spread.fact_id)
                {
                    self.last_search_results.push(spread.clone());
                }
            }

            // Re-sort by combined score and truncate
            self.last_search_results.sort_by(|a, b| {
                let score_a = a.similarity * 0.6 + a.confidence * 0.4;
                let score_b = b.similarity * 0.6 + b.confidence * 0.4;
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.last_search_results
                .truncate(self.config.search_top_k * 2);
        }

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
            // Epistemic surprise: contradictions + novelty
            epistemic_surprise: ((self.last_telemetry.contradictions_detected as f64 * 0.3)
                .min(0.6)
                + (1.0 - avg_sim) * 0.4)
                .clamp(0.0, 1.0),
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
            #[cfg(feature = "therapeutic")]
            EntityType::ClinicalConcept | EntityType::Symptom | EntityType::Intervention => {
                return Some("clinical".to_string());
            }
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
        let _ = mgr.ontology().total_queries(); // smoke: accessor doesn't panic
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

    // ── Lifecycle integration tests (hardening) ──

    /// Full lifecycle: 100 cycles of processing, verify telemetry accumulates correctly
    /// and all signals remain finite/bounded.
    #[test]
    fn test_100_cycle_lifecycle() {
        let config = KnowledgeManagerConfig {
            decay_interval: 10,
            ..Default::default()
        };
        let mut mgr = KnowledgeManager::new(config);
        mgr.bootstrap_entities();

        let inputs = [
            "The United States imposed sanctions on Russia.",
            "Russia retaliated with energy export restrictions.",
            "Oil prices increased globally due to supply constraints.",
            "The European Union diversified energy sources.",
            "China increased trade with Russia.",
            "Inflation rose in the United States.",
            "The Federal Reserve raised the interest rate.",
            "A recession was predicted by economists.",
            "Climate change accelerated deforestation in the Amazon.",
            "The pandemic caused supply chain disruptions.",
        ];

        for cycle in 0..100 {
            let input = inputs[cycle % inputs.len()];
            let (telem, signals) = mgr.process(input, cycle as u64);

            // All telemetry fields should be finite
            assert!(
                telem.avg_confidence.is_finite(),
                "cycle {cycle}: avg_confidence not finite"
            );
            assert!(
                telem.best_search_similarity.is_finite(),
                "cycle {cycle}: best_search_similarity not finite"
            );

            // All signals bounded
            assert!(
                signals.uncertainty >= 0.0 && signals.uncertainty <= 1.0,
                "cycle {cycle}: uncertainty out of bounds: {}",
                signals.uncertainty
            );
            assert!(
                signals.relevance >= 0.0 && signals.relevance <= 1.0,
                "cycle {cycle}: relevance out of bounds: {}",
                signals.relevance
            );
            assert!(
                signals.novelty >= 0.0 && signals.novelty <= 1.0,
                "cycle {cycle}: novelty out of bounds: {}",
                signals.novelty
            );
            assert!(
                signals.contradiction_signal.is_finite(),
                "cycle {cycle}: contradiction_signal not finite"
            );
            assert!(
                signals.causal_depth.is_finite(),
                "cycle {cycle}: causal_depth not finite"
            );
            assert!(
                signals.epistemic_surprise.is_finite(),
                "cycle {cycle}: epistemic_surprise not finite"
            );
        }

        // Graph should have accumulated facts
        assert!(
            mgr.graph().len() > 0,
            "Graph should have facts after 100 cycles"
        );

        // Telemetry graph_size should match
        let telem = mgr.telemetry();
        assert_eq!(telem.graph_size, mgr.graph().len() as u32);
    }

    /// Process empty input: should not panic, should not add facts
    #[test]
    fn test_empty_input_processing() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        let graph_before = mgr.graph().len();
        mgr.process("", 1);

        let telem = mgr.telemetry();
        assert_eq!(telem.facts_extracted, 0);
        assert_eq!(mgr.graph().len(), graph_before);
        assert!(mgr.signals().uncertainty.is_finite());
    }

    /// Contradiction detection: contradictory facts should produce alerts
    #[test]
    fn test_contradiction_alert_pipeline() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        // Insert similar but "contradictory" facts (same entities, negation)
        mgr.process("Sanctions reduced oil supply.", 1);
        mgr.process("Sanctions did not reduce oil supply.", 2);

        // Drain alerts — may or may not trigger depending on HDC similarity
        let alerts = mgr.drain_alerts();
        // Regardless, drain should work without panic
        let _ = alerts.len();

        // Second drain should be empty
        let alerts2 = mgr.drain_alerts();
        assert!(alerts2.is_empty(), "Second drain should be empty");
    }

    /// Calibration audit: record predictions and verify ECE/MCE are bounded
    #[test]
    fn test_calibration_audit_bounds() {
        let mut audit = CalibrationAudit::default();

        // Record 100 predictions with varying confidence and correctness
        for i in 0..100 {
            let confidence = (i as f32) / 100.0;
            let correct = i % 3 != 0; // 2/3 correct
            audit.record(confidence, correct);
        }

        let ece = audit.ece();
        let mce = audit.mce();

        assert!(ece >= 0.0 && ece <= 1.0, "ECE should be in [0,1]: {ece}");
        assert!(mce >= 0.0 && mce <= 1.0, "MCE should be in [0,1]: {mce}");
        assert!(audit.total_samples() == 100);

        let summary = audit.bin_summary();
        assert!(!summary.is_empty(), "Should have non-empty bins");
        for (midpoint, count, accuracy) in &summary {
            assert!(*midpoint >= 0.0 && *midpoint <= 1.0);
            assert!(*count > 0);
            assert!(*accuracy >= 0.0 && *accuracy <= 1.0);
        }
    }

    /// Calibration audit: empty audit should return 0
    #[test]
    fn test_calibration_audit_empty() {
        let audit = CalibrationAudit::default();
        assert_eq!(audit.ece(), 0.0);
        assert_eq!(audit.mce(), 0.0);
        assert_eq!(audit.total_samples(), 0);
        assert!(audit.bin_summary().is_empty());
    }

    #[test]
    fn test_calibration_audit_perfect() {
        let mut audit = CalibrationAudit::default();

        // Perfectly calibrated: 80% confidence facts are correct 80% of the time
        for _ in 0..80 {
            audit.record(0.85, true);
        }
        for _ in 0..20 {
            audit.record(0.85, false);
        }

        assert_eq!(audit.total_samples(), 100);
        // ECE should be low for near-perfect calibration
        assert!(audit.ece() < 0.15);
    }

    #[test]
    fn test_calibration_audit_overconfident() {
        let mut audit = CalibrationAudit::default();

        // Overconfident: 90% confidence but only 50% correct
        for _ in 0..50 {
            audit.record(0.95, true);
        }
        for _ in 0..50 {
            audit.record(0.95, false);
        }

        // ECE should be high (overconfident by ~0.4)
        assert!(audit.ece() > 0.3);
    }

    #[test]
    fn test_calibration_audit_bin_boundaries() {
        let mut audit = CalibrationAudit::default();

        audit.record(0.0, false); // Bin 0
        audit.record(0.15, true); // Bin 1
        audit.record(0.55, true); // Bin 5
        audit.record(0.99, true); // Bin 9

        let summary = audit.bin_summary();
        assert_eq!(summary.len(), 4); // 4 non-empty bins
        assert_eq!(audit.total_samples(), 4);
    }

    /// Consolidation: prune + strengthen should work on populated graph
    #[test]
    fn test_consolidate_and_forget() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        // Insert several facts
        for i in 0..10 {
            mgr.process(
                &format!("Event {i} caused consequence {i} in the United States."),
                i as u64,
            );
        }

        let graph_before = mgr.graph().len();
        let (pruned, strengthened) = mgr.consolidate_and_forget();

        // Should not panic
        let _ = pruned + strengthened;
        // Graph size should not increase after consolidation
        assert!(mgr.graph().len() <= graph_before);
    }

    /// Query interface: should return well-formed results
    #[test]
    fn test_query_interface() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        mgr.process("Climate change caused severe flooding.", 1);
        mgr.process("Flooding destroyed infrastructure.", 2);

        let result = mgr.query("climate flooding");
        // Result should have finite confidence values
        for fact in &result.facts {
            assert!(fact.confidence.is_finite());
            assert!(fact.similarity.is_finite());
        }
    }

    /// Processing interval: facts only extracted on-interval
    #[test]
    fn test_processing_interval() {
        let config = KnowledgeManagerConfig {
            processing_interval: 3,
            ..Default::default()
        };
        let mut mgr = KnowledgeManager::new(config);
        mgr.bootstrap_entities();

        // Cycle 0: on interval (0 % 3 == 0), should extract
        mgr.process("Sanctions caused oil shortage.", 0);
        let after_0 = mgr.graph().len();

        // Cycle 1: off interval, should skip extraction (but still search)
        mgr.process("New fact about inflation.", 1);
        let after_1 = mgr.graph().len();

        // Cycle 3: on interval again
        mgr.process("Another fact about trade.", 3);
        let after_3 = mgr.graph().len();

        assert!(after_0 > 0, "Should extract on cycle 0");
        // Off-interval shouldn't add new facts
        assert_eq!(after_1, after_0, "Should not extract on off-interval cycle");
        assert!(
            after_3 >= after_0,
            "Should extract on cycle 3 (on interval)"
        );
    }
}
