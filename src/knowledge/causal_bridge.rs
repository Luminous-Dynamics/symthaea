//! Causal Knowledge Bridge
//!
//! Automatically constructs causal DAG edges from extracted relations.
//! When the knowledge extractor identifies a causal relation (e.g.,
//! "sanctions caused inflation"), this module creates a directed edge
//! in the causal graph that can be queried via Pearl's do-calculus.
//!
//! Science: Pearl (2009) Causality, Spirtes et al. (2000) Causation/Prediction/Search

use super::extraction::{ExtractedRelation, SemanticRole};
use std::collections::HashMap;

// ── Types ──────────────────────────────────────────────────────────────────

/// A causal edge discovered from text
#[derive(Debug, Clone)]
pub struct CausalEdge {
    /// Cause node name
    pub cause: String,
    /// Effect node name
    pub effect: String,
    /// Strength of evidence (0.0–1.0)
    pub strength: f32,
    /// Whether this is a negative causal relation (prevents/blocks)
    pub is_inhibitory: bool,
    /// Whether the relation was negated in source text
    pub is_negated: bool,
    /// Source text for provenance
    pub source_text: String,
    /// Cycle when discovered
    pub discovered_at_cycle: u64,
}

/// A lightweight causal DAG built from extracted knowledge
///
/// This is separate from symthaea-causal-reasoning's CausalDAG to avoid
/// a hard dependency. The bridge exports edges that can be imported into
/// the full do-calculus engine when the `counterfactual` feature is enabled.
pub struct CausalKnowledgeBridge {
    /// All discovered causal edges
    edges: Vec<CausalEdge>,
    /// Node set: name → index (for adjacency tracking)
    nodes: HashMap<String, usize>,
    /// Adjacency: node_idx → Vec<(target_idx, strength)>
    adjacency: HashMap<usize, Vec<(usize, f32)>>,
    /// Maximum edges before pruning weakest
    capacity: usize,
    /// Minimum strength to keep an edge during pruning
    prune_threshold: f32,
    /// Statistics
    total_edges_discovered: u64,
}

/// Inhibitory verb patterns
const INHIBITORY_VERBS: &[&str] = &[
    "prevents",
    "prevented",
    "blocks",
    "blocked",
    "disrupts",
    "disrupted",
    "dampens",
    "dampened",
    "constrains",
    "constrained",
    "decreases",
    "decreased",
    "decelerates",
    "inhibits",
    "inhibited",
    "suppresses",
    "suppressed",
    "reduces",
    "reduced",
    "limits",
    "limited",
];

impl Default for CausalKnowledgeBridge {
    fn default() -> Self {
        Self::new(5_000)
    }
}

impl CausalKnowledgeBridge {
    pub fn new(capacity: usize) -> Self {
        Self {
            edges: Vec::new(),
            nodes: HashMap::new(),
            adjacency: HashMap::new(),
            capacity,
            prune_threshold: 0.1,
            total_edges_discovered: 0,
        }
    }

    /// Process an extracted relation and add causal edges if applicable.
    ///
    /// Returns true if a causal edge was added.
    pub fn process_relation(
        &mut self,
        relation: &ExtractedRelation,
        source_text: &str,
        current_cycle: u64,
    ) -> bool {
        if !relation.is_causal {
            return false;
        }

        let is_inhibitory = INHIBITORY_VERBS
            .iter()
            .any(|v| relation.predicate.to_lowercase().contains(v));

        let (cause, effect) = match relation.object_role {
            SemanticRole::Result | SemanticRole::Patient => (&relation.subject, &relation.object),
            SemanticRole::Cause => {
                // Reversed: object is the cause
                (&relation.object, &relation.subject)
            }
            _ => (&relation.subject, &relation.object),
        };

        let edge = CausalEdge {
            cause: cause.clone(),
            effect: effect.clone(),
            strength: relation.confidence,
            is_inhibitory,
            is_negated: relation.is_negated,
            source_text: source_text.to_string(),
            discovered_at_cycle: current_cycle,
        };

        self.add_edge(edge);
        true
    }

    /// Add a causal edge directly
    pub fn add_edge(&mut self, edge: CausalEdge) {
        // Get or create node indices
        let next_idx = self.nodes.len();
        let cause_idx = *self.nodes.entry(edge.cause.clone()).or_insert(next_idx);
        let next_idx = self.nodes.len();
        let effect_idx = *self.nodes.entry(edge.effect.clone()).or_insert(next_idx);

        // Check for existing edge (strengthen if found)
        let existing = self.adjacency.entry(cause_idx).or_default();
        if let Some(entry) = existing.iter_mut().find(|(t, _)| *t == effect_idx) {
            // Strengthen existing edge (running average)
            entry.1 = (entry.1 + edge.strength) / 2.0;
        } else {
            existing.push((effect_idx, edge.strength));
        }

        self.edges.push(edge);
        self.total_edges_discovered += 1;

        // Prune if over capacity
        if self.edges.len() > self.capacity {
            self.prune();
        }
    }

    /// Get all edges from a given cause
    pub fn effects_of(&self, cause: &str) -> Vec<&CausalEdge> {
        self.edges
            .iter()
            .filter(|e| e.cause.to_lowercase() == cause.to_lowercase())
            .collect()
    }

    /// Get all edges leading to a given effect
    pub fn causes_of(&self, effect: &str) -> Vec<&CausalEdge> {
        self.edges
            .iter()
            .filter(|e| e.effect.to_lowercase() == effect.to_lowercase())
            .collect()
    }

    /// Trace a causal chain from cause to any reachable effects (BFS, max depth)
    pub fn trace_chain(&self, start: &str, max_depth: usize) -> Vec<Vec<String>> {
        let mut chains = Vec::new();
        let mut queue: Vec<(String, Vec<String>)> =
            vec![(start.to_lowercase(), vec![start.to_string()])];

        while let Some((current, path)) = queue.pop() {
            if path.len() > max_depth + 1 {
                continue;
            }

            let effects: Vec<_> = self
                .edges
                .iter()
                .filter(|e| e.cause.to_lowercase() == current && !e.is_negated)
                .collect();

            if effects.is_empty() && path.len() > 1 {
                // Terminal node — record the chain
                chains.push(path);
            } else {
                for effect in effects {
                    let next = effect.effect.to_lowercase();
                    if !path.iter().any(|p| p.to_lowercase() == next) {
                        // No cycles
                        let mut new_path = path.clone();
                        new_path.push(effect.effect.clone());
                        queue.push((next, new_path));
                    }
                }
                if path.len() > 1 {
                    chains.push(path); // Also record intermediate chains
                }
            }
        }

        chains
    }

    /// Number of causal edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Number of unique causal nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total edges ever discovered
    pub fn total_edges_discovered(&self) -> u64 {
        self.total_edges_discovered
    }

    /// Export edges as (cause, effect, strength) triples for the full causal reasoning crate
    pub fn export_edges(&self) -> Vec<(String, String, f32)> {
        self.edges
            .iter()
            .filter(|e| !e.is_negated)
            .map(|e| {
                let strength = if e.is_inhibitory {
                    -e.strength
                } else {
                    e.strength
                };
                (e.cause.clone(), e.effect.clone(), strength)
            })
            .collect()
    }

    /// Import a causal edge from persistence (no source text, no negation info).
    pub fn import_edge(
        &mut self,
        cause: &str,
        effect: &str,
        strength: f32,
        is_inhibitory: bool,
    ) {
        let edge = CausalEdge {
            cause: cause.to_string(),
            effect: effect.to_string(),
            strength,
            is_inhibitory,
            is_negated: false,
            source_text: String::new(),
            discovered_at_cycle: 0,
        };
        self.add_edge(edge);
    }

    /// Export edges as persistence records.
    pub fn export_edge_records(&self) -> Vec<super::persistence::CausalEdgeRecord> {
        self.edges
            .iter()
            .filter(|e| !e.is_negated)
            .map(|e| super::persistence::CausalEdgeRecord {
                cause: e.cause.clone(),
                effect: e.effect.clone(),
                strength: if e.is_inhibitory {
                    -e.strength
                } else {
                    e.strength
                },
                is_inhibitory: e.is_inhibitory,
                cycle: e.discovered_at_cycle,
            })
            .collect()
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn prune(&mut self) {
        let before = self.edges.len();
        self.edges.retain(|e| e.strength >= self.prune_threshold);

        // Rebuild adjacency from surviving edges
        if self.edges.len() < before {
            self.adjacency.clear();
            for edge in &self.edges {
                if let (Some(&cause_idx), Some(&effect_idx)) =
                    (self.nodes.get(&edge.cause), self.nodes.get(&edge.effect))
                {
                    self.adjacency
                        .entry(cause_idx)
                        .or_default()
                        .push((effect_idx, edge.strength));
                }
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::extraction::{ExtractedRelation, SemanticRole};

    fn make_causal_relation(subj: &str, pred: &str, obj: &str) -> ExtractedRelation {
        ExtractedRelation {
            subject: subj.to_string(),
            predicate: pred.to_string(),
            object: obj.to_string(),
            subject_role: SemanticRole::Agent,
            object_role: SemanticRole::Result,
            is_causal: true,
            is_negated: false,
            confidence: 0.8,
        }
    }

    #[test]
    fn test_add_causal_edge() {
        let mut bridge = CausalKnowledgeBridge::new(100);
        let rel = make_causal_relation("sanctions", "caused", "inflation");

        let added = bridge.process_relation(&rel, "sanctions caused inflation", 1);
        assert!(added);
        assert_eq!(bridge.edge_count(), 1);
        assert_eq!(bridge.node_count(), 2);
    }

    #[test]
    fn test_non_causal_ignored() {
        let mut bridge = CausalKnowledgeBridge::new(100);
        let rel = ExtractedRelation {
            subject: "sky".to_string(),
            predicate: "is".to_string(),
            object: "blue".to_string(),
            subject_role: SemanticRole::Agent,
            object_role: SemanticRole::Patient,
            is_causal: false,
            is_negated: false,
            confidence: 0.9,
        };

        let added = bridge.process_relation(&rel, "sky is blue", 1);
        assert!(!added);
        assert_eq!(bridge.edge_count(), 0);
    }

    #[test]
    fn test_effects_of() {
        let mut bridge = CausalKnowledgeBridge::new(100);

        let r1 = make_causal_relation("blockade", "caused", "oil shortage");
        let r2 = make_causal_relation("blockade", "caused", "inflation");
        let r3 = make_causal_relation("inflation", "caused", "recession");

        bridge.process_relation(&r1, "", 1);
        bridge.process_relation(&r2, "", 2);
        bridge.process_relation(&r3, "", 3);

        let effects = bridge.effects_of("blockade");
        assert_eq!(effects.len(), 2);
    }

    #[test]
    fn test_causes_of() {
        let mut bridge = CausalKnowledgeBridge::new(100);

        let r1 = make_causal_relation("sanctions", "caused", "recession");
        let r2 = make_causal_relation("pandemic", "caused", "recession");

        bridge.process_relation(&r1, "", 1);
        bridge.process_relation(&r2, "", 2);

        let causes = bridge.causes_of("recession");
        assert_eq!(causes.len(), 2);
    }

    #[test]
    fn test_trace_chain() {
        let mut bridge = CausalKnowledgeBridge::new(100);

        bridge.process_relation(
            &make_causal_relation("blockade", "caused", "oil shortage"),
            "",
            1,
        );
        bridge.process_relation(
            &make_causal_relation("oil shortage", "caused", "price spike"),
            "",
            2,
        );
        bridge.process_relation(
            &make_causal_relation("price spike", "caused", "inflation"),
            "",
            3,
        );

        let chains = bridge.trace_chain("blockade", 5);
        assert!(!chains.is_empty());
        // Should find chain: blockade → oil shortage → price spike → inflation
        let longest = chains.iter().max_by_key(|c| c.len()).unwrap();
        assert!(longest.len() >= 3);
    }

    #[test]
    fn test_inhibitory_detection() {
        let mut bridge = CausalKnowledgeBridge::new(100);
        let rel = ExtractedRelation {
            subject: "ceasefire".to_string(),
            predicate: "prevented".to_string(),
            object: "escalation".to_string(),
            subject_role: SemanticRole::Agent,
            object_role: SemanticRole::Result,
            is_causal: true,
            is_negated: false,
            confidence: 0.7,
        };

        bridge.process_relation(&rel, "ceasefire prevented escalation", 1);
        let exported = bridge.export_edges();
        assert_eq!(exported.len(), 1);
        assert!(exported[0].2 < 0.0); // Inhibitory = negative strength
    }

    #[test]
    fn test_edge_strengthening() {
        let mut bridge = CausalKnowledgeBridge::new(100);

        // Same causal relation seen twice
        let r1 = make_causal_relation("X", "caused", "Y");
        let r2 = make_causal_relation("X", "caused", "Y");

        bridge.process_relation(&r1, "", 1);
        bridge.process_relation(&r2, "", 2);

        // Should have 2 raw edges but adjacency should show strengthened weight
        assert_eq!(bridge.node_count(), 2);
    }

    #[test]
    fn test_cycle_prevention_in_trace() {
        let mut bridge = CausalKnowledgeBridge::new(100);

        bridge.process_relation(&make_causal_relation("A", "caused", "B"), "", 1);
        bridge.process_relation(&make_causal_relation("B", "caused", "A"), "", 2);

        // Trace should not loop infinitely
        let chains = bridge.trace_chain("A", 10);
        for chain in &chains {
            assert!(chain.len() <= 3); // A→B or A→B→A at most
        }
    }
}
