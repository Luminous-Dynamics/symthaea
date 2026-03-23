// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # SelfSchema — Autobiographical Self-Model in the Knowledge System
//!
//! Provides a persistent, queryable self-model that bridges the Soul module's
//! identity/values with the KnowledgeManager's fact graph. The system can
//! query "historically, when I attempt X, do I succeed?" — enabling genuine
//! autobiographical memory rather than a scalar readout.
//!
//! # Architecture
//!
//! ```text
//! Soul (identity, values, experience) ──► SelfSchema ──► KnowledgeGraph
//!          ↑                                    │           (domain: "self")
//! FbSelfModel (meta_cognitive_accuracy,         │
//!   narrative_self_psi, embodied_agency)        ▼
//!                                          self_prototype (BinaryHV)
//!                                          narrative_coherence
//!                                          trait_stability
//! ```
//!
//! # Science
//!
//! - Northoff (2011): Default mode network activates during self-referential
//!   processing at ~10-40 second intervals.
//! - Linville (1985): Self-complexity ~30 distinct self-aspects.
//! - Roberts & DelVecchio (2000): Personality trait stability correlation ~0.98/year.
//! - Gallagher (2000): Narrative self requires temporal coherence above baseline.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use symthaea_core::hdc::BinaryHV;

/// Maximum self-facts in the schema before oldest are pruned.
/// Science: Linville (1985) — ~30 self-aspects × ~16 facts each ≈ 500.
pub const SELF_SCHEMA_MAX_FACTS: usize = 500;

/// EMA alpha for trait stability tracking (slow — personality is stable).
/// Science: Roberts & DelVecchio (2000) — trait stability ~0.98/year.
pub const SELF_SCHEMA_TRAIT_EMA_ALPHA: f32 = 0.05;

/// Minimum narrative coherence for self-continuity.
/// Science: Gallagher (2000) — narrative self requires temporal coherence.
pub const SELF_SCHEMA_NARRATIVE_THRESHOLD: f32 = 0.7;

/// Maximum trait history entries for stability computation.
pub const SELF_SCHEMA_TRAIT_HISTORY_CAP: usize = 32;

/// A self-referential fact — something the system knows about itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelfFactKind {
    /// Capability assertion: "I can do X" with confidence
    Capability { name: String, confidence: f32 },
    /// Limitation acknowledgment: "I cannot do X"
    Limitation { name: String },
    /// Trait observation: "I tend to X" with strength
    Trait { description: String, strength: f32 },
    /// Episode summary: "I experienced X at cycle Y"
    Episode {
        summary: String,
        valence: f32,
        cycle: u64,
    },
    /// Performance fact: "My accuracy for X is Y"
    Performance { domain: String, accuracy: f32 },
}

impl SelfFactKind {
    /// Short label for telemetry/logging.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Capability { .. } => "capability",
            Self::Limitation { .. } => "limitation",
            Self::Trait { .. } => "trait",
            Self::Episode { .. } => "episode",
            Self::Performance { .. } => "performance",
        }
    }

    /// Encode this fact as a text string suitable for HDC encoding.
    pub fn to_encoding_text(&self) -> String {
        match self {
            Self::Capability { name, confidence } => {
                format!("self can {} confidence {:.2}", name, confidence)
            }
            Self::Limitation { name } => format!("self cannot {}", name),
            Self::Trait {
                description,
                strength,
            } => {
                format!("self tends {} strength {:.2}", description, strength)
            }
            Self::Episode {
                summary,
                valence,
                cycle,
            } => format!(
                "self experienced {} valence {:.2} cycle {}",
                summary, valence, cycle
            ),
            Self::Performance { domain, accuracy } => {
                format!("self accuracy {} {:.2}", domain, accuracy)
            }
        }
    }
}

/// A stored self-fact with metadata.
#[derive(Debug, Clone)]
pub struct SelfFact {
    /// The fact content
    pub kind: SelfFactKind,
    /// HDC encoding of this fact
    pub encoding: BinaryHV,
    /// Cycle when this fact was recorded
    pub recorded_at_cycle: u64,
    /// How many times this fact has been reinforced
    pub reinforcement_count: u32,
}

/// Consolidated self-knowledge, updated periodically by SoulManager.
///
/// Maintains a running HDC prototype of "self" and tracks narrative
/// coherence (how stable the self-model is across updates).
pub struct SelfSchema {
    /// All stored self-facts (bounded by SELF_SCHEMA_MAX_FACTS)
    facts: VecDeque<SelfFact>,
    /// HDC prototype of "self" — running bundle of all self-fact encodings
    self_prototype: BinaryHV,
    /// Whether the prototype needs recomputation (dirty flag)
    prototype_dirty: bool,
    /// History of prototype cosine similarities for narrative coherence
    coherence_history: VecDeque<f32>,
    /// Current narrative coherence (EMA of consecutive prototype similarities)
    narrative_coherence: f32,
    /// Trait stability: how much the self-model changes per update (low = stable)
    trait_stability: f32,
    /// Last update cycle
    last_update_cycle: u64,
    /// Total facts ever inserted (including pruned)
    total_facts_inserted: u64,
}

impl SelfSchema {
    /// Create a new empty self-schema.
    pub fn new() -> Self {
        Self {
            facts: VecDeque::new(),
            self_prototype: BinaryHV::random(0x53454C46), // "SELF" in ASCII hex
            prototype_dirty: false,
            coherence_history: VecDeque::new(),
            narrative_coherence: 1.0, // starts coherent (no history to diverge from)
            trait_stability: 1.0,     // starts stable
            last_update_cycle: 0,
            total_facts_inserted: 0,
        }
    }

    /// Insert a new self-fact.
    ///
    /// The fact is HDC-encoded via its text representation and stored.
    /// If at capacity, the oldest fact is pruned (FIFO).
    pub fn insert(&mut self, kind: SelfFactKind, cycle: u64) {
        let text = kind.to_encoding_text();
        // Simple deterministic encoding: hash the text to get a seed, then create BinaryHV
        let seed = Self::text_seed(&text);
        let encoding = BinaryHV::random(seed);

        let fact = SelfFact {
            kind,
            encoding,
            recorded_at_cycle: cycle,
            reinforcement_count: 1,
        };

        // Check for similar existing fact (reinforcement instead of duplicate)
        let mut reinforced = false;
        for existing in self.facts.iter_mut() {
            if existing.encoding.similarity(&fact.encoding) > 0.8 {
                existing.reinforcement_count += 1;
                existing.recorded_at_cycle = cycle;
                reinforced = true;
                break;
            }
        }

        if !reinforced {
            if self.facts.len() >= SELF_SCHEMA_MAX_FACTS {
                self.facts.pop_front(); // FIFO pruning
            }
            self.facts.push_back(fact);
            self.total_facts_inserted += 1;
        }

        self.prototype_dirty = true;
        self.last_update_cycle = cycle;
    }

    /// Query self-knowledge by HDC similarity.
    ///
    /// Returns the top-k most similar self-facts to the query vector.
    pub fn query(&self, query: &BinaryHV, top_k: usize) -> Vec<&SelfFact> {
        let mut scored: Vec<(&SelfFact, f32)> = self
            .facts
            .iter()
            .map(|f| (f, f.encoding.similarity(query)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).map(|(f, _)| f).collect()
    }

    /// Recompute the self-prototype from all facts and update coherence.
    ///
    /// Should be called periodically (e.g., on SoulManager tick) after
    /// facts have been inserted.
    pub fn recompute_prototype(&mut self) {
        if !self.prototype_dirty || self.facts.is_empty() {
            return;
        }

        let old_prototype = self.self_prototype.clone();

        // Bundle all fact encodings into a new prototype
        let encodings: Vec<BinaryHV> = self.facts.iter().map(|f| f.encoding.clone()).collect();
        self.self_prototype = BinaryHV::bundle(&encodings);

        // Compute narrative coherence: similarity between old and new prototype
        let sim = old_prototype.similarity(&self.self_prototype);
        let sim = if sim.is_finite() { sim } else { 0.0 };

        self.coherence_history.push_back(sim);
        if self.coherence_history.len() > SELF_SCHEMA_TRAIT_HISTORY_CAP {
            self.coherence_history.pop_front();
        }

        // Update narrative coherence EMA
        self.narrative_coherence = self.narrative_coherence * (1.0 - SELF_SCHEMA_TRAIT_EMA_ALPHA)
            + sim * SELF_SCHEMA_TRAIT_EMA_ALPHA;
        if !self.narrative_coherence.is_finite() {
            self.narrative_coherence = 0.5;
        }

        // Trait stability: 1.0 - change_rate (high similarity = high stability)
        self.trait_stability = self.trait_stability * (1.0 - SELF_SCHEMA_TRAIT_EMA_ALPHA)
            + sim * SELF_SCHEMA_TRAIT_EMA_ALPHA;
        if !self.trait_stability.is_finite() {
            self.trait_stability = 0.5;
        }

        self.prototype_dirty = false;
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// The HDC prototype of "self" — a running bundle of all self-facts.
    pub fn self_prototype(&self) -> &BinaryHV {
        &self.self_prototype
    }

    /// Current narrative coherence (0.0-1.0).
    /// High = stable self-model; low = identity in flux.
    pub fn narrative_coherence(&self) -> f32 {
        self.narrative_coherence
    }

    /// Current trait stability (0.0-1.0).
    /// High = personality traits are stable; low = rapid personality change.
    pub fn trait_stability(&self) -> f32 {
        self.trait_stability
    }

    /// Number of stored self-facts.
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    /// Total facts ever inserted (including pruned).
    pub fn total_facts_inserted(&self) -> u64 {
        self.total_facts_inserted
    }

    /// Last cycle at which a fact was inserted.
    pub fn last_update_cycle(&self) -> u64 {
        self.last_update_cycle
    }

    /// Whether narrative coherence is above the continuity threshold.
    pub fn has_narrative_continuity(&self) -> bool {
        self.narrative_coherence >= SELF_SCHEMA_NARRATIVE_THRESHOLD
    }

    /// Telemetry snapshot.
    pub fn telemetry(&self) -> SelfSchemaTelemetry {
        SelfSchemaTelemetry {
            fact_count: self.facts.len() as u32,
            narrative_coherence: self.narrative_coherence,
            trait_stability: self.trait_stability,
            total_facts_inserted: self.total_facts_inserted,
            has_narrative_continuity: self.has_narrative_continuity(),
        }
    }

    // ── Internal ───────────────────────────────────────────────────────

    /// Deterministic seed from text (simple hash for reproducibility).
    fn text_seed(text: &str) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset
        for byte in text.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3); // FNV-1a prime
        }
        hash
    }
}

impl Default for SelfSchema {
    fn default() -> Self {
        Self::new()
    }
}

/// Telemetry for SelfSchema, reported in CycleMetadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SelfSchemaTelemetry {
    pub fact_count: u32,
    pub narrative_coherence: f32,
    pub trait_stability: f32,
    pub total_facts_inserted: u64,
    pub has_narrative_continuity: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Use a fixed seed constructor for BinaryHV::random
    // (the real implementation takes a seed parameter)

    #[test]
    fn test_insert_and_count() {
        let mut schema = SelfSchema::new();
        assert_eq!(schema.fact_count(), 0);

        schema.insert(
            SelfFactKind::Capability {
                name: "moral reasoning".into(),
                confidence: 0.8,
            },
            100,
        );
        assert_eq!(schema.fact_count(), 1);
        assert_eq!(schema.last_update_cycle(), 100);
    }

    #[test]
    fn test_capacity_pruning() {
        let mut schema = SelfSchema::new();
        for i in 0..SELF_SCHEMA_MAX_FACTS + 10 {
            schema.insert(
                SelfFactKind::Performance {
                    domain: format!("domain_{}", i),
                    accuracy: 0.5,
                },
                i as u64,
            );
        }
        assert!(schema.fact_count() <= SELF_SCHEMA_MAX_FACTS);
    }

    #[test]
    fn test_reinforcement_dedup() {
        let mut schema = SelfSchema::new();
        // Insert same fact twice — should reinforce, not duplicate
        schema.insert(
            SelfFactKind::Capability {
                name: "reasoning".into(),
                confidence: 0.7,
            },
            100,
        );
        schema.insert(
            SelfFactKind::Capability {
                name: "reasoning".into(),
                confidence: 0.7,
            },
            200,
        );
        // Should be 1 fact with reinforcement_count = 2
        assert_eq!(schema.fact_count(), 1);
    }

    #[test]
    fn test_narrative_coherence_initial() {
        let schema = SelfSchema::new();
        assert_eq!(schema.narrative_coherence(), 1.0); // starts coherent
        assert!(schema.has_narrative_continuity());
    }

    #[test]
    fn test_recompute_prototype_updates_coherence() {
        let mut schema = SelfSchema::new();
        schema.insert(
            SelfFactKind::Trait {
                description: "curious".into(),
                strength: 0.8,
            },
            100,
        );
        schema.recompute_prototype();

        // After first recompute, coherence should still be reasonable
        assert!(schema.narrative_coherence() > 0.0);
        assert!(schema.trait_stability() > 0.0);
    }

    #[test]
    fn test_query_returns_results() {
        let mut schema = SelfSchema::new();
        schema.insert(
            SelfFactKind::Capability {
                name: "language generation".into(),
                confidence: 0.9,
            },
            100,
        );
        schema.insert(
            SelfFactKind::Limitation {
                name: "physical embodiment".into(),
            },
            101,
        );

        // Query with the prototype itself should return results
        let results = schema.query(schema.self_prototype(), 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_telemetry() {
        let mut schema = SelfSchema::new();
        schema.insert(
            SelfFactKind::Performance {
                domain: "ethics".into(),
                accuracy: 0.85,
            },
            50,
        );

        let telem = schema.telemetry();
        assert_eq!(telem.fact_count, 1);
        assert!(telem.narrative_coherence > 0.0);
        assert!(telem.has_narrative_continuity);
    }

    #[test]
    fn test_self_fact_kind_encoding_text() {
        let cap = SelfFactKind::Capability {
            name: "reasoning".into(),
            confidence: 0.8,
        };
        assert!(cap.to_encoding_text().contains("self can reasoning"));

        let lim = SelfFactKind::Limitation {
            name: "flying".into(),
        };
        assert!(lim.to_encoding_text().contains("self cannot flying"));
    }

    #[test]
    fn test_nan_guards() {
        let mut schema = SelfSchema::new();
        // Force a NaN-producing scenario by inserting then clearing
        schema.narrative_coherence = f32::NAN;
        schema.insert(
            SelfFactKind::Trait {
                description: "test".into(),
                strength: 0.5,
            },
            1,
        );
        schema.recompute_prototype();
        // NaN guard should have kicked in
        assert!(schema.narrative_coherence().is_finite());
        assert!(schema.trait_stability().is_finite());
    }
}
