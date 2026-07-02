// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Sacred Stillness: Known Unknowns Module
//!
//! Implements epistemic humility — explicitly modeling what the system
//! does NOT know, rather than only tracking what it does know.
//!
//! The 8th Harmony (Sacred Stillness / Apophatic-Knowing) represents
//! the void from which all arises. In epistemic terms, this is the
//! space of acknowledged unknowns that constrain and inform knowledge.
//!
//! ## How It Works
//!
//! When the knowledge graph fails to resolve a query, the system
//! registers the gap as a `KnownUnknown`. These accumulate into
//! an `apophatic_depth` score that modulates Broca's confidence —
//! generating in domains with known unknowns produces more hedged,
//! careful language.
//!
//! ## Science
//!
//! - Socratic ignorance: "I know that I know nothing"
//! - Rumsfeld matrix: known-knowns, known-unknowns, unknown-unknowns
//! - Goldberg (2023): Epistemic immunity degrades without active maintenance
//! - The Contemplative epistemic context (E=0.20, N=0.30, M=0.50) naturally
//!   weights permanence/depth highest, aligning with apophatic traditions

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A recognized gap in knowledge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnknownDimension {
    /// Knowledge domain (e.g., "quantum_gravity", "social_dynamics")
    pub domain: String,
    /// How precisely the unknown is characterized (0.0 = vague, 1.0 = sharp boundary)
    pub specificity: f64,
    /// How much this unknown matters for current tasks (0.0 = irrelevant, 1.0 = critical)
    pub importance: f64,
    /// Comfort with not-knowing (0.0 = anxious/rushing to fill, 1.0 = serene acceptance)
    pub acceptance: f64,
    /// Cycle when this unknown was first registered
    pub registered_at: u64,
    /// Number of times this unknown was encountered
    pub encounter_count: u64,
}

/// Tracks the system's known unknowns — epistemic humility as computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownUnknowns {
    /// Map from domain to its unknown dimension
    unknowns: HashMap<String, UnknownDimension>,
    /// Aggregate apophatic depth: how deeply we know what we don't know (0.0-1.0)
    apophatic_depth: f64,
    /// Maximum unknowns to track (prevents unbounded growth)
    capacity: usize,
}

/// Named constant: default capacity for known unknowns registry
const KNOWN_UNKNOWNS_CAPACITY: usize = 128;

/// Named constant: importance decay per 1000 cycles (unknowns that aren't
/// re-encountered become less important over time)
const IMPORTANCE_DECAY_PER_THOUSAND: f64 = 0.1;

/// Named constant: maximum confidence reduction from known unknowns (30%)
const MAX_CONFIDENCE_REDUCTION: f64 = 0.30;

impl Default for KnownUnknowns {
    fn default() -> Self {
        Self {
            unknowns: HashMap::new(),
            apophatic_depth: 0.0,
            capacity: KNOWN_UNKNOWNS_CAPACITY,
        }
    }
}

impl KnownUnknowns {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a knowledge gap. If the domain already exists, update it.
    pub fn register_unknown(
        &mut self,
        domain: &str,
        importance: f64,
        specificity: f64,
        cycle: u64,
    ) {
        let importance = importance.clamp(0.0, 1.0);
        let specificity = specificity.clamp(0.0, 1.0);

        if let Some(existing) = self.unknowns.get_mut(domain) {
            existing.encounter_count += 1;
            // Importance increases with repeated encounters
            existing.importance = (existing.importance + importance * 0.1).clamp(0.0, 1.0);
            // Specificity sharpens with repeated encounters
            existing.specificity = (existing.specificity + specificity * 0.05).clamp(0.0, 1.0);
        } else {
            // Evict least important if at capacity
            if self.unknowns.len() >= self.capacity {
                if let Some(min_key) = self
                    .unknowns
                    .iter()
                    .min_by(|a, b| a.1.importance.total_cmp(&b.1.importance))
                    .map(|(k, _)| k.clone())
                {
                    self.unknowns.remove(&min_key);
                }
            }

            self.unknowns.insert(
                domain.to_string(),
                UnknownDimension {
                    domain: domain.to_string(),
                    specificity,
                    importance,
                    acceptance: 0.5, // Start neutral
                    registered_at: cycle,
                    encounter_count: 1,
                },
            );
        }

        self.recompute_apophatic_depth();
    }

    /// Remove a known unknown (domain has been learned about).
    pub fn resolve_unknown(&mut self, domain: &str) {
        self.unknowns.remove(domain);
        self.recompute_apophatic_depth();
    }

    /// Decay importance of unknowns that haven't been re-encountered.
    pub fn decay_importance(&mut self, current_cycle: u64) {
        let decay_rate = IMPORTANCE_DECAY_PER_THOUSAND / 1000.0;
        self.unknowns.retain(|_, dim| {
            let age = current_cycle.saturating_sub(dim.registered_at) as f64;
            dim.importance -= decay_rate * age;
            dim.importance = dim.importance.max(0.0);
            dim.importance > 0.001 // Remove negligible unknowns
        });
        self.recompute_apophatic_depth();
    }

    /// Modulate generation confidence based on known unknowns in a domain.
    ///
    /// Returns a multiplier (0.7-1.0): domains with important unknowns
    /// reduce confidence, encouraging hedged/careful language.
    pub fn modulate_confidence(&self, domain: &str) -> f64 {
        if let Some(dim) = self.unknowns.get(domain) {
            let reduction = dim.importance * dim.specificity * MAX_CONFIDENCE_REDUCTION;
            1.0 - reduction
        } else {
            1.0 // No known unknowns in this domain
        }
    }

    /// Aggregate importance of all known unknowns.
    pub fn total_importance(&self) -> f64 {
        self.unknowns.values().map(|d| d.importance).sum()
    }

    /// Current apophatic depth (0.0-1.0).
    pub fn apophatic_depth(&self) -> f64 {
        self.apophatic_depth
    }

    /// Number of tracked unknowns.
    pub fn count(&self) -> usize {
        self.unknowns.len()
    }

    /// Get a specific unknown dimension.
    pub fn get(&self, domain: &str) -> Option<&UnknownDimension> {
        self.unknowns.get(domain)
    }

    /// All tracked unknown domains.
    pub fn domains(&self) -> Vec<&str> {
        self.unknowns.keys().map(|s| s.as_str()).collect()
    }

    fn recompute_apophatic_depth(&mut self) {
        if self.unknowns.is_empty() {
            self.apophatic_depth = 0.0;
            return;
        }
        // Apophatic depth = mean(specificity * importance * acceptance)
        // High depth means: we know precisely what we don't know, it matters,
        // and we're at peace with the gap.
        let sum: f64 = self
            .unknowns
            .values()
            .map(|d| d.specificity * d.importance * d.acceptance)
            .sum();
        self.apophatic_depth = (sum / self.unknowns.len() as f64).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_retrieve() {
        let mut ku = KnownUnknowns::new();
        ku.register_unknown("quantum_gravity", 0.8, 0.3, 100);
        assert_eq!(ku.count(), 1);
        assert!(ku.get("quantum_gravity").is_some());
        assert_eq!(ku.get("quantum_gravity").unwrap().encounter_count, 1);
    }

    #[test]
    fn test_repeated_encounter_increases_importance() {
        let mut ku = KnownUnknowns::new();
        ku.register_unknown("dark_matter", 0.5, 0.5, 100);
        let initial = ku.get("dark_matter").unwrap().importance;
        ku.register_unknown("dark_matter", 0.5, 0.5, 200);
        let after = ku.get("dark_matter").unwrap().importance;
        assert!(after > initial);
        assert_eq!(ku.get("dark_matter").unwrap().encounter_count, 2);
    }

    #[test]
    fn test_modulate_confidence_unknown_domain() {
        let mut ku = KnownUnknowns::new();
        ku.register_unknown("speculative_physics", 0.9, 0.8, 100);
        let conf = ku.modulate_confidence("speculative_physics");
        assert!(conf < 1.0, "confidence should be reduced: {conf}");
        assert!(conf >= 0.7, "reduction should be bounded: {conf}");
    }

    #[test]
    fn test_modulate_confidence_known_domain() {
        let ku = KnownUnknowns::new();
        let conf = ku.modulate_confidence("well_understood");
        assert_eq!(conf, 1.0);
    }

    #[test]
    fn test_resolve_unknown() {
        let mut ku = KnownUnknowns::new();
        ku.register_unknown("mystery", 0.5, 0.5, 100);
        assert_eq!(ku.count(), 1);
        ku.resolve_unknown("mystery");
        assert_eq!(ku.count(), 0);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut ku = KnownUnknowns::default();
        let cap = ku.capacity;
        for i in 0..cap + 5 {
            ku.register_unknown(
                &format!("domain_{i}"),
                (i as f64) / (cap as f64),
                0.5,
                i as u64,
            );
        }
        assert_eq!(ku.count(), cap);
    }

    #[test]
    fn test_apophatic_depth_empty() {
        let ku = KnownUnknowns::new();
        assert_eq!(ku.apophatic_depth(), 0.0);
    }

    #[test]
    fn test_decay_removes_negligible() {
        let mut ku = KnownUnknowns::new();
        ku.register_unknown("fading", 0.01, 0.5, 0);
        ku.decay_importance(100_000);
        assert_eq!(ku.count(), 0);
    }
}
