// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Adaptive Ontology — Primitive Learning System
//!
//! Enables the HDC primitive system to grow new concepts from experience.
//! When Symthaea encounters a concept that doesn't map well to existing
//! primitives, a new derived composite is created. Usage frequency and
//! reasoning utility are tracked, enabling Hebbian strengthening of
//! useful primitives and pruning of unused ones.
//!
//! Science: Quine (1960) ontological relativity, Carey (2009) conceptual change,
//!          Hebb (1949) organization of behavior

use std::collections::HashMap;
use symthaea_core::hdc::unified_hv::{BinaryHV, HDC_DIMENSION};

use crate::knowledge::persistence::OntologyRecord;

// ── Types ──────────────────────────────────────────────────────────────────

/// Usage statistics for a learned primitive
#[derive(Debug, Clone)]
pub struct PrimitiveUsage {
    /// Name of the primitive
    pub name: String,
    /// The learned HDC vector
    pub vector: BinaryHV,
    /// Number of times used in encoding/retrieval
    pub usage_count: u64,
    /// Cumulative reasoning utility (how much did using this primitive
    /// improve prediction accuracy or reduce prediction error?)
    pub utility: f64,
    /// Cycle when this primitive was created
    pub created_at_cycle: u64,
    /// Cycle when last used
    pub last_used_cycle: u64,
    /// Whether this primitive was composed from existing ones
    pub parent_names: Vec<String>,
    /// IS-A parent concept name (e.g., "dog" is_a "animal").
    /// Science: Quillian (1967) — semantic networks; Collins & Loftus (1975).
    pub is_a_parent: Option<String>,
}

/// Configuration for the adaptive ontology
#[derive(Debug, Clone)]
pub struct AdaptiveOntologyConfig {
    /// Maximum number of learned primitives
    pub max_primitives: usize,
    /// Minimum similarity to consider an existing primitive a match
    pub match_threshold: f32,
    /// Minimum usage count before a primitive is eligible for pruning
    pub prune_min_age_cycles: u64,
    /// Minimum utility for a primitive to survive pruning
    pub prune_min_utility: f64,
    /// How often to run pruning (cycles)
    pub prune_interval: u64,
    /// Hebbian learning rate for strengthening useful primitives
    pub hebbian_rate: f32,
}

impl Default for AdaptiveOntologyConfig {
    fn default() -> Self {
        Self {
            max_primitives: 500,
            match_threshold: 0.6,
            prune_min_age_cycles: 1000,
            prune_min_utility: 0.01,
            prune_interval: 500,
            hebbian_rate: 0.01,
        }
    }
}

// ── Adaptive Ontology ──────────────────────────────────────────────────────

/// Manages the growing set of learned primitives
pub struct AdaptiveOntology {
    /// Configuration
    config: AdaptiveOntologyConfig,
    /// All learned primitives
    primitives: HashMap<String, PrimitiveUsage>,
    /// Last pruning cycle
    last_prune_cycle: u64,
    /// Statistics
    total_created: u64,
    total_pruned: u64,
    total_queries: u64,
}

impl Default for AdaptiveOntology {
    fn default() -> Self {
        Self::new(AdaptiveOntologyConfig::default())
    }
}

impl AdaptiveOntology {
    pub fn new(config: AdaptiveOntologyConfig) -> Self {
        Self {
            config,
            primitives: HashMap::new(),
            last_prune_cycle: 0,
            total_created: 0,
            total_pruned: 0,
            total_queries: 0,
        }
    }

    /// Look up the best matching primitive for a given concept vector.
    ///
    /// Returns the primitive name and similarity if above threshold.
    /// Increments usage count on match.
    pub fn lookup(&mut self, concept: &BinaryHV, current_cycle: u64) -> Option<(&str, f32)> {
        self.total_queries += 1;

        let mut best_name = None;
        let mut best_sim = -1.0_f32;

        for (name, usage) in &self.primitives {
            let sim = concept.similarity(&usage.vector);
            if sim > best_sim {
                best_sim = sim;
                best_name = Some(name.clone());
            }
        }

        if best_sim >= self.config.match_threshold {
            if let Some(ref name) = best_name {
                if let Some(usage) = self.primitives.get_mut(name) {
                    usage.usage_count += 1;
                    usage.last_used_cycle = current_cycle;
                }
                if let Some(primitive) = self.primitives.get(name) {
                    return Some((primitive.name.as_str(), best_sim));
                }
            }
        }

        None
    }

    /// Learn a new primitive from a concept that didn't match existing ones.
    ///
    /// Returns true if a new primitive was created, false if it was
    /// merged with an existing one or rejected (at capacity).
    pub fn learn(
        &mut self,
        name: &str,
        vector: BinaryHV,
        parent_names: Vec<String>,
        current_cycle: u64,
    ) -> bool {
        // Check if we already have something similar
        let mut best_sim = 0.0_f32;
        let mut best_match = None;
        for (existing_name, usage) in &self.primitives {
            let sim = vector.similarity(&usage.vector);
            if sim > best_sim {
                best_sim = sim;
                best_match = Some(existing_name.clone());
            }
        }

        if best_sim > self.config.match_threshold {
            // Similar primitive exists — strengthen via Hebbian update
            if let Some(ref match_name) = best_match {
                if let Some(usage) = self.primitives.get_mut(match_name) {
                    // Hebbian: move existing vector slightly toward new concept
                    // (In binary HDC, this is approximated by majority vote)
                    let old = usage.vector;
                    usage.vector = BinaryHV::bundle(&[old, vector]);
                    usage.usage_count += 1;
                    usage.last_used_cycle = current_cycle;
                }
            }
            return false;
        }

        // At capacity — need to evict or reject
        if self.primitives.len() >= self.config.max_primitives {
            // Evict lowest utility primitive
            let lowest = self
                .primitives
                .iter()
                .filter(|(_, u)| {
                    current_cycle.saturating_sub(u.created_at_cycle)
                        > self.config.prune_min_age_cycles
                })
                .min_by(|(_, a), (_, b)| {
                    a.utility
                        .partial_cmp(&b.utility)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(k, _)| k.clone());

            if let Some(key) = lowest {
                self.primitives.remove(&key);
                self.total_pruned += 1;
            } else {
                return false; // All primitives are too young to evict
            }
        }

        // Create new primitive
        let usage = PrimitiveUsage {
            name: name.to_string(),
            vector,
            usage_count: 1,
            utility: 0.0,
            created_at_cycle: current_cycle,
            last_used_cycle: current_cycle,
            parent_names,
            is_a_parent: None,
        };

        self.primitives.insert(name.to_string(), usage);
        self.total_created += 1;
        true
    }

    /// Set the IS-A parent for a primitive (taxonomic relation).
    ///
    /// Example: `set_is_a("dog", "animal")` means "dog IS-A animal".
    /// Science: Quillian (1967) — semantic networks; Collins & Loftus (1975).
    pub fn set_is_a(&mut self, child: &str, parent: &str) {
        if let Some(usage) = self.primitives.get_mut(child) {
            usage.is_a_parent = Some(parent.to_string());
        }
    }

    /// Record utility feedback for a primitive.
    ///
    /// Called after a primitive was used in reasoning and we know how
    /// much it helped (positive = useful, negative = misleading).
    pub fn record_utility(&mut self, name: &str, utility_delta: f64) {
        if let Some(usage) = self.primitives.get_mut(name) {
            usage.utility += utility_delta;
        }
    }

    /// Periodic pruning: remove low-utility, old, unused primitives.
    ///
    /// Returns number of primitives pruned.
    pub fn maybe_prune(&mut self, current_cycle: u64) -> usize {
        if current_cycle.saturating_sub(self.last_prune_cycle) < self.config.prune_interval {
            return 0;
        }
        self.last_prune_cycle = current_cycle;

        let before = self.primitives.len();

        self.primitives.retain(|_, usage| {
            let age = current_cycle.saturating_sub(usage.created_at_cycle);
            if age < self.config.prune_min_age_cycles {
                return true; // Too young to prune
            }
            // Keep if utility is above threshold or if recently used
            let recently_used = current_cycle.saturating_sub(usage.last_used_cycle)
                < self.config.prune_min_age_cycles;
            usage.utility >= self.config.prune_min_utility || recently_used
        });

        let pruned = before - self.primitives.len();
        self.total_pruned += pruned as u64;
        pruned
    }

    /// Walk the IS-A parent chain starting from `concept`, up to `max_depth` steps.
    ///
    /// Returns a vector of ancestors, e.g. `["animal", "living_thing"]` for `"dog"`.
    /// Returns an empty vector if the concept is not found or has no IS-A parent.
    ///
    /// Science: Collins & Quillian (1969) — hierarchical semantic memory retrieval.
    pub fn is_a_chain(&self, concept: &str, max_depth: usize) -> Vec<String> {
        let mut chain = Vec::new();
        let mut current = concept.to_string();
        for _ in 0..max_depth {
            match self
                .primitives
                .get(&current)
                .and_then(|u| u.is_a_parent.as_ref())
            {
                Some(parent) => {
                    chain.push(parent.clone());
                    current = parent.clone();
                }
                None => break,
            }
        }
        chain
    }

    /// Return all concepts that have `parent` as their direct IS-A parent.
    ///
    /// Science: Collins & Quillian (1969) — bidirectional traversal of taxonomic hierarchies.
    pub fn children_of(&self, parent: &str) -> Vec<String> {
        self.primitives
            .iter()
            .filter_map(|(name, usage)| {
                if usage.is_a_parent.as_deref() == Some(parent) {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check whether `child` transitively IS-A `ancestor`.
    ///
    /// Walks the IS-A parent chain from `child` up to 20 steps looking for `ancestor`.
    ///
    /// Science: Collins & Quillian (1969) — spreading activation in semantic networks.
    pub fn is_a(&self, child: &str, ancestor: &str) -> bool {
        let chain = self.is_a_chain(child, 20);
        chain.contains(&ancestor.to_string())
    }

    /// Get all learned primitives (for inspection/telemetry)
    pub fn primitives(&self) -> &HashMap<String, PrimitiveUsage> {
        &self.primitives
    }

    /// Number of learned primitives
    pub fn count(&self) -> usize {
        self.primitives.len()
    }

    /// Total primitives ever created
    pub fn total_created(&self) -> u64 {
        self.total_created
    }

    /// Total primitives ever pruned
    pub fn total_pruned(&self) -> u64 {
        self.total_pruned
    }

    /// Total lookup queries
    pub fn total_queries(&self) -> u64 {
        self.total_queries
    }

    /// Average utility across all learned primitives
    pub fn average_utility(&self) -> f64 {
        if self.primitives.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.primitives.values().map(|u| u.utility).sum();
        sum / self.primitives.len() as f64
    }

    /// Export all primitives as persistence records for SQLite storage.
    pub fn export_ontology_records(&self) -> Vec<OntologyRecord> {
        self.primitives
            .values()
            .map(|u| OntologyRecord {
                name: u.name.clone(),
                vector_bytes: u.vector.0.to_vec(),
                usage_count: u.usage_count,
                utility: u.utility,
                created_at_cycle: u.created_at_cycle,
                last_used_cycle: u.last_used_cycle,
                is_a_parent: u.is_a_parent.clone(),
            })
            .collect()
    }

    /// Import a primitive from a persistence record loaded from SQLite.
    ///
    /// Silently skips records whose vector bytes are the wrong length.
    pub fn import_ontology_record(&mut self, record: &OntologyRecord) {
        if record.vector_bytes.len() < BinaryHV::BYTES {
            return;
        }
        let mut arr = [0u8; BinaryHV::BYTES];
        arr.copy_from_slice(&record.vector_bytes[..BinaryHV::BYTES]);
        let vector = BinaryHV(arr);
        // Use `learn` to insert without overwriting existing primitives
        // (it merges via Hebbian update if a similar primitive already exists).
        self.learn(&record.name, vector, Vec::new(), record.created_at_cycle);
        // Restore persisted statistics if the primitive was inserted (not merged).
        if let Some(u) = self.primitives.get_mut(&record.name) {
            u.usage_count = u.usage_count.max(record.usage_count);
            u.utility = record.utility;
            u.last_used_cycle = record.last_used_cycle;
            if record.is_a_parent.is_some() {
                u.is_a_parent = record.is_a_parent.clone();
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learn_new_primitive() {
        let mut ontology = AdaptiveOntology::default();
        let vec = BinaryHV::random(42);

        let created = ontology.learn("new_concept", vec, vec![], 1);
        assert!(created);
        assert_eq!(ontology.count(), 1);
        assert_eq!(ontology.total_created(), 1);
    }

    #[test]
    fn test_lookup_matching() {
        let mut ontology = AdaptiveOntology::default();
        let vec = BinaryHV::random(42);

        ontology.learn("my_concept", vec.clone(), vec![], 1);

        // Lookup with the same vector should match
        let result = ontology.lookup(&vec, 2);
        assert!(result.is_some());
        let (name, sim) = result.unwrap();
        assert_eq!(name, "my_concept");
        assert!(sim > 0.9);
    }

    #[test]
    fn test_lookup_no_match() {
        let mut ontology = AdaptiveOntology::default();
        let vec1 = BinaryHV::random(42);
        let vec2 = BinaryHV::random(999);

        ontology.learn("concept_a", vec1, vec![], 1);

        // Different vector should not match
        let result = ontology.lookup(&vec2, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_hebbian_strengthening() {
        let mut ontology = AdaptiveOntology::default();
        let vec1 = BinaryHV::random(42);

        ontology.learn("concept", vec1.clone(), vec![], 1);

        // Learning a very similar vector should strengthen, not create new
        let created = ontology.learn("concept_v2", vec1.clone(), vec![], 2);
        assert!(!created); // Merged, not created
        assert_eq!(ontology.count(), 1);
    }

    #[test]
    fn test_utility_tracking() {
        let mut ontology = AdaptiveOntology::default();
        let vec = BinaryHV::random(42);

        ontology.learn("useful_concept", vec, vec![], 1);
        ontology.record_utility("useful_concept", 0.5);
        ontology.record_utility("useful_concept", 0.3);

        let prim = ontology.primitives().get("useful_concept").unwrap();
        assert!((prim.utility - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_pruning() {
        let config = AdaptiveOntologyConfig {
            max_primitives: 100,
            prune_min_age_cycles: 10,
            prune_min_utility: 0.1,
            prune_interval: 5,
            ..Default::default()
        };
        let mut ontology = AdaptiveOntology::new(config);

        // Create a low-utility primitive
        let vec = BinaryHV::random(42);
        ontology.learn("useless", vec, vec![], 1);
        // Don't record any utility

        // Prune after enough cycles
        let pruned = ontology.maybe_prune(50);
        assert_eq!(pruned, 1);
        assert_eq!(ontology.count(), 0);
    }

    #[test]
    fn test_capacity_eviction() {
        let config = AdaptiveOntologyConfig {
            max_primitives: 3,
            prune_min_age_cycles: 0, // Allow immediate eviction
            ..Default::default()
        };
        let mut ontology = AdaptiveOntology::new(config);

        for i in 0..5 {
            let vec = BinaryHV::random(i as u64 * 1000);
            ontology.learn(&format!("concept_{i}"), vec, vec![], i as u64);
        }

        assert!(ontology.count() <= 3);
    }

    #[test]
    fn test_parent_tracking() {
        let mut ontology = AdaptiveOntology::default();
        let vec = BinaryHV::random(42);

        ontology.learn(
            "composite",
            vec,
            vec!["parent_a".to_string(), "parent_b".to_string()],
            1,
        );

        let prim = ontology.primitives().get("composite").unwrap();
        assert_eq!(prim.parent_names.len(), 2);
    }

    #[test]
    fn test_average_utility() {
        let mut ontology = AdaptiveOntology::default();

        let v1 = BinaryHV::random(42);
        let v2 = BinaryHV::random(999);
        ontology.learn("a", v1, vec![], 1);
        ontology.learn("b", v2, vec![], 1);

        ontology.record_utility("a", 1.0);
        ontology.record_utility("b", 0.5);

        assert!((ontology.average_utility() - 0.75).abs() < 0.001);
    }
}
