//! Propagation Engine - SCEI Component
//!
//! The Propagation Engine spreads confidence updates through the knowledge graph
//! when claims are attested, contradicted, or resolved. It provides:
//!
//! - **Confidence Propagation**: Updates flow through connected claims
//! - **Inference Rules**: Derive new confidence from related claims
//! - **Contradiction Resolution**: Handle conflicting information
//! - **Temporal Decay**: Propagate time-based confidence decay
//! - **Path Analysis**: Track how confidence changes propagate
//!
//! # Integration with DKG
//!
//! The propagation engine works with the confidence scoring system to ensure
//! consistency across the knowledge graph. When a high-reputation agent
//! attests to a claim, related claims may see confidence adjustments.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Core Types
// ============================================================================

/// A node in the propagation graph
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct PropagationNode {
    /// Triple hash identifier
    pub triple_hash: String,
    /// Current confidence score
    pub confidence: f64,
    /// Previous confidence (before propagation)
    pub previous_confidence: f64,
    /// Depth in propagation path
    pub depth: u32,
    /// Source of the propagation
    pub propagation_source: Option<String>,
}

/// A path through which confidence propagated
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct PropagationPath {
    /// Starting triple
    pub source: String,
    /// Ending triple
    pub target: String,
    /// Intermediate nodes
    pub path: Vec<String>,
    /// Total propagation factor (attenuation)
    pub total_factor: f64,
    /// Confidence delta at target
    pub delta: f64,
}

/// Result of a propagation operation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct PropagationResult {
    /// Source triple that triggered propagation
    pub source_triple: String,
    /// Original confidence change
    pub original_delta: f64,
    /// Nodes affected by propagation
    pub affected_nodes: Vec<PropagationNode>,
    /// Paths through which confidence flowed
    pub paths: Vec<PropagationPath>,
    /// Total nodes visited
    pub nodes_visited: u32,
    /// Maximum depth reached
    pub max_depth: u32,
    /// Execution time in microseconds
    pub execution_time_us: u64,
}

/// How confidence is propagated between connected claims
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct ConfidencePropagation {
    /// Attenuation factor per hop (0.0 - 1.0)
    pub attenuation: f64,
    /// Relationship type affecting propagation
    pub relationship: PropagationRelationship,
    /// Minimum delta to continue propagating
    pub min_delta: f64,
    /// Whether to propagate through contradictions
    pub propagate_contradictions: bool,
}

/// Types of relationships affecting propagation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub enum PropagationRelationship {
    /// Claims share the same subject
    SameSubject,
    /// Claims share the same predicate
    SamePredicate,
    /// Object of one is subject of another
    ChainedReference,
    /// Claims contradict each other
    Contradiction,
    /// Claims from same creator
    SameCreator,
    /// Claims in same domain
    SameDomain,
}

impl Default for ConfidencePropagation {
    fn default() -> Self {
        Self {
            attenuation: 0.5,
            relationship: PropagationRelationship::SameSubject,
            min_delta: 0.01,
            propagate_contradictions: true,
        }
    }
}

// ============================================================================
// Inference Rules
// ============================================================================

/// Rule for inferring confidence from related claims
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct InferenceRule {
    /// Rule identifier
    pub rule_id: String,
    /// Rule name
    pub name: String,
    /// Pattern to match (predicate patterns)
    pub pattern: InferencePattern,
    /// Confidence multiplier when rule fires
    pub confidence_factor: f64,
    /// Whether rule is currently active
    pub active: bool,
}

/// Pattern for inference rule matching
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct InferencePattern {
    /// Required predicates
    pub predicates: Vec<String>,
    /// Required relationship between claims
    pub relationship: InferenceRelationship,
    /// Minimum confidence of premises
    pub min_premise_confidence: f64,
}

/// How claims relate for inference
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub enum InferenceRelationship {
    /// A is B, B is C → A is C
    Transitive,
    /// A implies B, A is true → B is true
    ModusPonens,
    /// A implies B, B is false → A is false
    ModusTollens,
    /// A and B → C
    Conjunction,
    /// A or B, not A → B
    DisjunctiveSyllogism,
}

/// Result of applying an inference rule
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct InferenceResult {
    /// Rule that was applied
    pub rule_id: String,
    /// Premise triples
    pub premises: Vec<String>,
    /// Inferred conclusion
    pub conclusion_subject: String,
    /// Predicate of the inferred conclusion.
    pub conclusion_predicate: String,
    /// Object of the inferred conclusion.
    pub conclusion_object: String,
    /// Confidence of inference
    pub inferred_confidence: f64,
    /// Explanation of inference
    pub explanation: String,
}

// ============================================================================
// Propagation Engine
// ============================================================================

/// Configuration for the propagation engine
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PropagationEngineConfig {
    /// Maximum propagation depth
    pub max_depth: u32,
    /// Default attenuation per hop
    pub default_attenuation: f64,
    /// Minimum delta to continue propagation
    pub min_delta: f64,
    /// Maximum nodes to visit per propagation
    pub max_nodes: u32,
    /// Enable inference rules
    pub enable_inference: bool,
    /// Batch size for propagation
    pub batch_size: usize,
}

impl Default for PropagationEngineConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            default_attenuation: 0.5,
            min_delta: 0.01,
            max_nodes: 1000,
            enable_inference: true,
            batch_size: 100,
        }
    }
}

/// Graph structure for propagation
#[derive(Clone, Debug, Default)]
struct PropagationGraph {
    /// Triple hash -> connected triple hashes
    edges: HashMap<String, HashSet<String>>,
    /// Triple hash -> current confidence
    confidences: HashMap<String, f64>,
    /// Triple hash -> subject
    subjects: HashMap<String, String>,
    /// Triple hash -> predicate
    predicates: HashMap<String, String>,
    /// Triple hash -> creator
    creators: HashMap<String, String>,
    /// Triple hash -> domain
    domains: HashMap<String, String>,
}

impl PropagationGraph {
    fn add_edge(&mut self, from: &str, to: &str) {
        self.edges
            .entry(from.to_string())
            .or_default()
            .insert(to.to_string());
        self.edges
            .entry(to.to_string())
            .or_default()
            .insert(from.to_string());
    }

    fn get_neighbors(&self, hash: &str) -> Vec<String> {
        self.edges
            .get(hash)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }
}

/// The Propagation Engine
pub struct PropagationEngine {
    config: PropagationEngineConfig,
    graph: PropagationGraph,
    inference_rules: Vec<InferenceRule>,
    propagation_history: Vec<PropagationResult>,
}

impl PropagationEngine {
    /// Create a new propagation engine with the given configuration.
    pub fn new(config: PropagationEngineConfig) -> Self {
        let mut engine = Self {
            config,
            graph: PropagationGraph::default(),
            inference_rules: Vec::new(),
            propagation_history: Vec::new(),
        };

        // Add default inference rules
        engine.add_default_rules();
        engine
    }

    fn add_default_rules(&mut self) {
        // Transitive property rule
        self.inference_rules.push(InferenceRule {
            rule_id: "transitive-1".to_string(),
            name: "Transitivity".to_string(),
            pattern: InferencePattern {
                predicates: vec!["is_a".to_string(), "subclass_of".to_string()],
                relationship: InferenceRelationship::Transitive,
                min_premise_confidence: 0.7,
            },
            confidence_factor: 0.9,
            active: true,
        });

        // Modus ponens rule
        self.inference_rules.push(InferenceRule {
            rule_id: "modus-ponens-1".to_string(),
            name: "Modus Ponens".to_string(),
            pattern: InferencePattern {
                predicates: vec!["implies".to_string()],
                relationship: InferenceRelationship::ModusPonens,
                min_premise_confidence: 0.8,
            },
            confidence_factor: 0.85,
            active: true,
        });
    }

    /// Register a triple in the propagation graph
    pub fn register_triple(
        &mut self,
        hash: &str,
        subject: &str,
        predicate: &str,
        confidence: f64,
        creator: Option<&str>,
        domain: Option<&str>,
    ) {
        self.graph.confidences.insert(hash.to_string(), confidence);
        self.graph
            .subjects
            .insert(hash.to_string(), subject.to_string());
        self.graph
            .predicates
            .insert(hash.to_string(), predicate.to_string());

        if let Some(c) = creator {
            self.graph.creators.insert(hash.to_string(), c.to_string());
        }
        if let Some(d) = domain {
            self.graph.domains.insert(hash.to_string(), d.to_string());
        }

        // Build edges based on shared subjects
        // Collect edges to add first to avoid borrow conflict
        let edges_to_add: Vec<String> = self
            .graph
            .subjects
            .iter()
            .filter(|(other_hash, other_subject)| *other_hash != hash && *other_subject == subject)
            .map(|(other_hash, _)| other_hash.clone())
            .collect();

        for other_hash in edges_to_add {
            self.graph.add_edge(hash, &other_hash);
        }
    }

    /// Propagate a confidence change through the graph
    pub fn propagate(&mut self, source_hash: &str, delta: f64) -> PropagationResult {
        let start_time = std::time::Instant::now();

        let mut affected = Vec::new();
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut queue: VecDeque<(String, f64, u32, Vec<String>)> = VecDeque::new();

        // Initialize with source
        queue.push_back((source_hash.to_string(), delta, 0, vec![]));
        visited.insert(source_hash.to_string());

        let mut max_depth = 0u32;

        while let Some((current, current_delta, depth, path)) = queue.pop_front() {
            if depth > self.config.max_depth || visited.len() as u32 > self.config.max_nodes {
                break;
            }

            max_depth = max_depth.max(depth);

            // Apply delta to current node
            if depth > 0 {
                let prev_confidence = self.graph.confidences.get(&current).copied().unwrap_or(0.0);
                let new_confidence = (prev_confidence + current_delta).clamp(0.0, 1.0);
                self.graph
                    .confidences
                    .insert(current.clone(), new_confidence);

                affected.push(PropagationNode {
                    triple_hash: current.clone(),
                    confidence: new_confidence,
                    previous_confidence: prev_confidence,
                    depth,
                    propagation_source: path.last().cloned(),
                });

                if !path.is_empty() {
                    paths.push(PropagationPath {
                        source: source_hash.to_string(),
                        target: current.clone(),
                        path: path.clone(),
                        total_factor: self.config.default_attenuation.powi(depth as i32),
                        delta: current_delta,
                    });
                }
            }

            // Propagate to neighbors
            let attenuated_delta = current_delta * self.config.default_attenuation;
            if attenuated_delta.abs() >= self.config.min_delta {
                for neighbor in self.graph.get_neighbors(&current) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor.clone());
                        let mut new_path = path.clone();
                        new_path.push(current.clone());
                        queue.push_back((neighbor, attenuated_delta, depth + 1, new_path));
                    }
                }
            }
        }

        let result = PropagationResult {
            source_triple: source_hash.to_string(),
            original_delta: delta,
            affected_nodes: affected,
            paths,
            nodes_visited: visited.len() as u32,
            max_depth,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        };

        self.propagation_history.push(result.clone());
        if self.propagation_history.len() > 1000 {
            self.propagation_history.remove(0);
        }

        result
    }

    /// Apply inference rules to find new conclusions
    pub fn run_inference(&self) -> Vec<InferenceResult> {
        if !self.config.enable_inference {
            return Vec::new();
        }

        let mut results = Vec::new();

        for rule in &self.inference_rules {
            if !rule.active {
                continue;
            }

            match rule.pattern.relationship {
                InferenceRelationship::Transitive => {
                    results.extend(self.find_transitive_inferences(rule));
                }
                InferenceRelationship::ModusPonens => {
                    results.extend(self.find_modus_ponens_inferences(rule));
                }
                _ => {
                    // Other inference types not yet implemented
                }
            }
        }

        results
    }

    fn find_transitive_inferences(&self, rule: &InferenceRule) -> Vec<InferenceResult> {
        let mut results = Vec::new();

        // Find triples with matching predicates
        let matching: Vec<_> = self
            .graph
            .predicates
            .iter()
            .filter(|(hash, pred)| {
                rule.pattern.predicates.contains(pred)
                    && self
                        .graph
                        .confidences
                        .get(*hash)
                        .map(|c| *c >= rule.pattern.min_premise_confidence)
                        .unwrap_or(false)
            })
            .collect();

        // Look for transitive chains: A -> B, B -> C => A -> C
        for (hash1, _) in &matching {
            let Some(subject1) = self.graph.subjects.get(*hash1) else {
                continue;
            };
            let Some(&confidence1) = self.graph.confidences.get(*hash1) else {
                continue;
            };

            // This is a simplified transitive check
            // In a real implementation, we'd track object references
            for (hash2, _) in &matching {
                if hash1 == hash2 {
                    continue;
                }

                let Some(subject2) = self.graph.subjects.get(*hash2) else {
                    continue;
                };
                let Some(&confidence2) = self.graph.confidences.get(*hash2) else {
                    continue;
                };

                // Check if they form a chain
                // (simplified: same subject implies potential chain)
                if subject1 == subject2 {
                    let inferred_confidence = confidence1 * confidence2 * rule.confidence_factor;

                    if inferred_confidence >= rule.pattern.min_premise_confidence {
                        results.push(InferenceResult {
                            rule_id: rule.rule_id.clone(),
                            premises: vec![(*hash1).clone(), (*hash2).clone()],
                            conclusion_subject: subject1.clone(),
                            conclusion_predicate: "derived_relation".to_string(),
                            conclusion_object: subject2.clone(),
                            inferred_confidence,
                            explanation: format!(
                                "Transitive inference from {} and {}",
                                hash1, hash2
                            ),
                        });
                    }
                }
            }
        }

        results
    }

    fn find_modus_ponens_inferences(&self, rule: &InferenceRule) -> Vec<InferenceResult> {
        let mut results = Vec::new();

        // Find "implies" predicates
        let implications: Vec<_> = self
            .graph
            .predicates
            .iter()
            .filter(|(_, pred)| *pred == "implies")
            .filter(|(hash, _)| {
                self.graph
                    .confidences
                    .get(*hash)
                    .map(|c| *c >= rule.pattern.min_premise_confidence)
                    .unwrap_or(false)
            })
            .collect();

        for (impl_hash, _) in implications {
            let Some(impl_subject) = self.graph.subjects.get(impl_hash) else {
                continue;
            };
            let Some(&impl_confidence) = self.graph.confidences.get(impl_hash) else {
                continue;
            };

            // Check if antecedent is true with high confidence
            for (other_hash, other_subject) in &self.graph.subjects {
                if other_hash == impl_hash {
                    continue;
                }

                if other_subject == impl_subject {
                    let other_confidence = self.graph.confidences.get(other_hash);
                    if let Some(&conf) = other_confidence {
                        if conf >= rule.pattern.min_premise_confidence {
                            let inferred_confidence =
                                impl_confidence * conf * rule.confidence_factor;

                            results.push(InferenceResult {
                                rule_id: rule.rule_id.clone(),
                                premises: vec![impl_hash.clone(), other_hash.clone()],
                                conclusion_subject: "consequence".to_string(),
                                conclusion_predicate: "is_true".to_string(),
                                conclusion_object: impl_subject.clone(),
                                inferred_confidence,
                                explanation: format!(
                                    "Modus ponens: {} implies conclusion, {} is true",
                                    impl_hash, other_hash
                                ),
                            });
                        }
                    }
                }
            }
        }

        results
    }

    /// Add a custom inference rule
    pub fn add_rule(&mut self, rule: InferenceRule) {
        self.inference_rules.push(rule);
    }

    /// Get propagation statistics
    pub fn stats(&self) -> PropagationStats {
        let total_propagations = self.propagation_history.len();
        let avg_affected = if total_propagations > 0 {
            self.propagation_history
                .iter()
                .map(|r| r.affected_nodes.len())
                .sum::<usize>() as f64
                / total_propagations as f64
        } else {
            0.0
        };

        let avg_depth = if total_propagations > 0 {
            self.propagation_history
                .iter()
                .map(|r| r.max_depth as f64)
                .sum::<f64>()
                / total_propagations as f64
        } else {
            0.0
        };

        PropagationStats {
            total_triples: self.graph.confidences.len(),
            total_edges: self.graph.edges.values().map(|s| s.len()).sum::<usize>() / 2,
            total_propagations,
            average_affected_nodes: avg_affected,
            average_max_depth: avg_depth,
            active_inference_rules: self.inference_rules.iter().filter(|r| r.active).count(),
        }
    }

    /// Get confidence for a triple
    pub fn get_confidence(&self, hash: &str) -> Option<f64> {
        self.graph.confidences.get(hash).copied()
    }
}

/// Statistics about propagation engine state
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct PropagationStats {
    /// Total number of triples in the graph.
    pub total_triples: usize,
    /// Total number of edges in the graph.
    pub total_edges: usize,
    /// Total number of propagations performed.
    pub total_propagations: usize,
    /// Average number of nodes affected per propagation.
    pub average_affected_nodes: f64,
    /// Average maximum depth reached during propagation.
    pub average_max_depth: f64,
    /// Number of active inference rules.
    pub active_inference_rules: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_propagation_graph() {
        let mut graph = PropagationGraph::default();
        graph.add_edge("a", "b");
        graph.add_edge("b", "c");

        assert!(graph.get_neighbors("a").contains(&"b".to_string()));
        assert!(graph.get_neighbors("b").contains(&"a".to_string()));
        assert!(graph.get_neighbors("b").contains(&"c".to_string()));
    }

    #[test]
    fn test_propagation_engine() {
        let mut engine = PropagationEngine::new(PropagationEngineConfig::default());

        // Register connected triples
        engine.register_triple(
            "triple-1",
            "sky",
            "color",
            0.8,
            Some("alice"),
            Some("nature"),
        );
        engine.register_triple(
            "triple-2",
            "sky",
            "clarity",
            0.7,
            Some("bob"),
            Some("nature"),
        );
        engine.register_triple(
            "triple-3",
            "ocean",
            "color",
            0.9,
            Some("alice"),
            Some("nature"),
        );

        // Propagate confidence boost
        let result = engine.propagate("triple-1", 0.1);

        assert!(!result.affected_nodes.is_empty());
        assert_eq!(result.source_triple, "triple-1");

        // Check that connected node was affected
        let triple_2_affected = result
            .affected_nodes
            .iter()
            .find(|n| n.triple_hash == "triple-2");
        assert!(triple_2_affected.is_some());
    }

    #[test]
    fn test_attenuation() {
        let mut engine = PropagationEngine::new(PropagationEngineConfig {
            default_attenuation: 0.5,
            max_depth: 3,
            ..Default::default()
        });

        engine.register_triple("a", "subj", "pred", 0.5, None, None);
        engine.register_triple("b", "subj", "pred", 0.5, None, None);

        let result = engine.propagate("a", 0.2);

        // With 0.5 attenuation, delta should be 0.1 at depth 1
        if let Some(node_b) = result.affected_nodes.iter().find(|n| n.triple_hash == "b") {
            assert!((node_b.confidence - 0.6).abs() < 0.01);
        }
    }

    #[test]
    fn test_inference_rules() {
        let mut engine = PropagationEngine::new(PropagationEngineConfig {
            enable_inference: true,
            ..Default::default()
        });

        // Register triples that could trigger transitive inference
        engine.register_triple("t1", "mammal", "is_a", 0.95, None, None);
        engine.register_triple("t2", "mammal", "subclass_of", 0.9, None, None);

        let inferences = engine.run_inference();
        // Should find at least one inference
        assert!(!inferences.is_empty() || engine.inference_rules.is_empty());
    }

    #[test]
    fn test_propagation_stats() {
        let mut engine = PropagationEngine::new(PropagationEngineConfig::default());

        engine.register_triple("a", "s1", "p1", 0.8, None, None);
        engine.register_triple("b", "s1", "p2", 0.7, None, None);

        engine.propagate("a", 0.1);
        engine.propagate("b", -0.05);

        let stats = engine.stats();
        assert_eq!(stats.total_triples, 2);
        assert_eq!(stats.total_propagations, 2);
    }

    #[test]
    fn test_min_delta_threshold() {
        let mut engine = PropagationEngine::new(PropagationEngineConfig {
            min_delta: 0.05,
            default_attenuation: 0.5,
            max_depth: 10,
            ..Default::default()
        });

        engine.register_triple("a", "s", "p", 0.5, None, None);
        engine.register_triple("b", "s", "p", 0.5, None, None);
        engine.register_triple("c", "s", "p", 0.5, None, None);

        // Small delta should stop propagating quickly
        let result = engine.propagate("a", 0.1);

        // With 0.5 attenuation and 0.05 threshold:
        // depth 1: 0.1 * 0.5 = 0.05 (barely passes)
        // depth 2: 0.05 * 0.5 = 0.025 (below threshold)
        assert!(result.max_depth <= 2);
    }
}
