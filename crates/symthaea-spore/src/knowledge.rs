// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Knowledge Engine — semantic knowledge graph for the Spore WASM kernel.
//!
//! Port of the desktop KnowledgeManager into a lightweight, WASM-compatible
//! module. Stores subject-predicate-object triples with confidence tracking,
//! contradiction detection, and temporal decay.
//!
//! Fully WASM-compatible: no std::time, no filesystem, no tokio.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum number of facts in the knowledge base.
const DEFAULT_MAX_FACTS: usize = 500;

/// Confidence decay rate per cycle for unaccessed facts.
/// After 100 cycles without access: 0.999^100 = ~0.905.
const CONFIDENCE_DECAY_RATE: f32 = 0.999;

/// Minimum confidence below which facts become candidates for forgetting.
const FORGET_THRESHOLD: f32 = 0.05;

/// Confidence boost on access (recency effect).
const ACCESS_BOOST: f32 = 0.01;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A fact with confidence and source tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeFact {
    /// Subject of the triple (e.g. "sun").
    pub subject: String,
    /// Predicate / relationship (e.g. "is_a").
    pub predicate: String,
    /// Object of the triple (e.g. "star").
    pub object: String,
    /// Confidence in this fact (0-1).
    pub confidence: f32,
    /// How this fact was acquired.
    pub source: String,
    /// Cycle at which this fact was learned.
    pub cycle_learned: u64,
    /// Number of times this fact has been accessed.
    pub access_count: u32,
}

/// Query result for WASM/JSON export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub facts: Vec<KnowledgeFact>,
    pub total_matches: usize,
}

/// Knowledge engine statistics for WASM/JSON export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeStats {
    pub fact_count: usize,
    pub max_facts: usize,
    pub contradiction_count: u32,
    pub total_queries: u64,
    pub total_learns: u64,
    pub avg_confidence: f32,
}

// ---------------------------------------------------------------------------
// KnowledgeEngine
// ---------------------------------------------------------------------------

/// Semantic knowledge graph with confidence tracking and contradiction detection.
pub struct KnowledgeEngine {
    facts: Vec<KnowledgeFact>,
    max_facts: usize,
    contradiction_count: u32,
    total_queries: u64,
    total_learns: u64,
    current_cycle: u64,
}

impl KnowledgeEngine {
    /// Create a new KnowledgeEngine with the given capacity.
    pub fn new(max_facts: usize) -> Self {
        Self {
            facts: Vec::with_capacity(max_facts.min(DEFAULT_MAX_FACTS)),
            max_facts: max_facts.max(1),
            contradiction_count: 0,
            total_queries: 0,
            total_learns: 0,
            current_cycle: 0,
        }
    }

    /// Create with default capacity.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_MAX_FACTS)
    }

    /// Learn a new fact. If a conflicting fact exists, records a contradiction.
    /// If the fact already exists with the same SPO, updates confidence.
    pub fn learn(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        source: &str,
        cycle: u64,
    ) {
        self.current_cycle = cycle;
        self.total_learns += 1;

        // Check for existing identical fact → boost confidence
        if let Some(existing) = self
            .facts
            .iter_mut()
            .find(|f| f.subject == subject && f.predicate == predicate && f.object == object)
        {
            existing.confidence = (existing.confidence + 0.1).clamp(0.0, 1.0);
            existing.access_count += 1;
            return;
        }

        // Check for contradictions: same subject+predicate but different object
        let has_contradiction = self
            .facts
            .iter()
            .any(|f| f.subject == subject && f.predicate == predicate && f.object != object);

        if has_contradiction {
            self.contradiction_count += 1;
        }

        // Evict weakest fact if at capacity
        if self.facts.len() >= self.max_facts {
            self.forget_weakest();
        }

        // Initial confidence based on source
        let confidence = match source {
            "observed" => 0.9,
            "inferred" => 0.6,
            "told" => 0.5,
            _ => 0.4,
        };

        self.facts.push(KnowledgeFact {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            confidence,
            source: source.to_string(),
            cycle_learned: cycle,
            access_count: 0,
        });
    }

    /// Query facts matching the given subject and predicate.
    /// Returns facts sorted by confidence (descending).
    pub fn query(&mut self, subject: &str, predicate: &str) -> QueryResult {
        self.total_queries += 1;

        let mut matches: Vec<KnowledgeFact> = self
            .facts
            .iter_mut()
            .filter(|f| f.subject == subject && f.predicate == predicate)
            .map(|f| {
                f.access_count += 1;
                // Boost confidence on access (recency effect)
                f.confidence = (f.confidence + ACCESS_BOOST).clamp(0.0, 1.0);
                f.clone()
            })
            .collect();

        let total_matches = matches.len();
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        QueryResult {
            facts: matches,
            total_matches,
        }
    }

    /// Query all facts about a subject.
    pub fn query_about(&mut self, subject: &str) -> QueryResult {
        self.total_queries += 1;

        let mut matches: Vec<KnowledgeFact> = self
            .facts
            .iter_mut()
            .filter(|f| f.subject == subject)
            .map(|f| {
                f.access_count += 1;
                f.confidence = (f.confidence + ACCESS_BOOST).clamp(0.0, 1.0);
                f.clone()
            })
            .collect();

        let total_matches = matches.len();
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        QueryResult {
            facts: matches,
            total_matches,
        }
    }

    /// Detect contradictions: facts with the same subject+predicate but different objects.
    /// Returns pairs of contradicting facts.
    pub fn detect_contradictions(&self) -> Vec<(KnowledgeFact, KnowledgeFact)> {
        let mut contradictions = Vec::new();

        for (i, a) in self.facts.iter().enumerate() {
            for b in self.facts.iter().skip(i + 1) {
                if a.subject == b.subject && a.predicate == b.predicate && a.object != b.object {
                    contradictions.push((a.clone(), b.clone()));
                }
            }
        }

        contradictions
    }

    /// Apply confidence decay to all facts that haven't been accessed recently.
    /// Call this periodically (e.g., every N cycles).
    pub fn confidence_decay(&mut self, elapsed_cycles: u64) {
        let decay = CONFIDENCE_DECAY_RATE.powi(elapsed_cycles as i32);
        for fact in &mut self.facts {
            // Only decay facts that haven't been accessed recently
            if fact.access_count == 0 || (self.current_cycle > fact.cycle_learned + 50) {
                fact.confidence *= decay;
            }
        }
    }

    /// Remove the lowest-confidence fact (when at capacity).
    pub fn forget_weakest(&mut self) {
        if self.facts.is_empty() {
            return;
        }

        let min_idx = self
            .facts
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.facts.remove(min_idx);
    }

    /// Prune all facts below the forget threshold.
    pub fn prune(&mut self) {
        self.facts.retain(|f| f.confidence >= FORGET_THRESHOLD);
    }

    /// Total number of facts currently stored.
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    /// Total number of contradictions detected since creation.
    pub fn contradiction_count(&self) -> u32 {
        self.contradiction_count
    }

    /// Export statistics for WASM/JSON serialization.
    pub fn stats(&self) -> KnowledgeStats {
        let avg_confidence = if self.facts.is_empty() {
            0.0
        } else {
            self.facts.iter().map(|f| f.confidence).sum::<f32>() / self.facts.len() as f32
        };

        KnowledgeStats {
            fact_count: self.facts.len(),
            max_facts: self.max_facts,
            contradiction_count: self.contradiction_count,
            total_queries: self.total_queries,
            total_learns: self.total_learns,
            avg_confidence,
        }
    }

    /// Update the current cycle counter (for decay calculations).
    pub fn set_cycle(&mut self, cycle: u64) {
        self.current_cycle = cycle;
    }
}

impl Default for KnowledgeEngine {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learn_and_query() {
        let mut engine = KnowledgeEngine::new(100);
        engine.learn("sun", "is_a", "star", "observed", 1);
        engine.learn("sun", "color", "yellow", "observed", 1);
        engine.learn("moon", "orbits", "earth", "told", 2);

        let result = engine.query("sun", "is_a");
        assert_eq!(result.total_matches, 1);
        assert_eq!(result.facts[0].object, "star");
        assert!(result.facts[0].confidence > 0.8);

        let about_sun = engine.query_about("sun");
        assert_eq!(about_sun.total_matches, 2);
    }

    #[test]
    fn test_contradiction_detection() {
        let mut engine = KnowledgeEngine::new(100);
        engine.learn("earth", "shape", "round", "observed", 1);
        engine.learn("earth", "shape", "flat", "told", 2);

        assert_eq!(engine.contradiction_count(), 1);

        let contradictions = engine.detect_contradictions();
        assert_eq!(contradictions.len(), 1);
        assert_eq!(contradictions[0].0.object, "round");
        assert_eq!(contradictions[0].1.object, "flat");
    }

    #[test]
    fn test_confidence_decay_and_pruning() {
        let mut engine = KnowledgeEngine::new(100);
        engine.learn("old_fact", "rel", "val", "told", 1);

        // Simulate 1000 cycles of decay
        engine.set_cycle(1001);
        engine.confidence_decay(1000);

        // Confidence should have decayed significantly
        assert!(engine.facts[0].confidence < 0.5);

        // After extreme decay, prune should remove it
        for _ in 0..5 {
            engine.confidence_decay(1000);
        }
        engine.prune();
        assert_eq!(engine.fact_count(), 0);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut engine = KnowledgeEngine::new(3);
        engine.learn("a", "r", "1", "told", 1);
        engine.learn("b", "r", "2", "told", 2);
        engine.learn("c", "r", "3", "observed", 3); // higher confidence

        // Fourth fact should evict the weakest (lowest confidence among a,b)
        engine.learn("d", "r", "4", "observed", 4);
        assert_eq!(engine.fact_count(), 3);
    }

    #[test]
    fn test_duplicate_fact_boosts_confidence() {
        let mut engine = KnowledgeEngine::new(100);
        engine.learn("x", "rel", "y", "told", 1);
        let initial = engine.facts[0].confidence;

        engine.learn("x", "rel", "y", "told", 2);
        assert!(engine.facts[0].confidence > initial);
        assert_eq!(engine.fact_count(), 1); // no duplicate
    }
}
