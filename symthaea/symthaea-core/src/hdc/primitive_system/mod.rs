// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Primitive System - Beyond NSM to Universal Ontological Primes
//!
//! **Revolutionary Architecture for Artificial Wisdom**
//!
//! While Natural Semantic Metalanguage (NSM) provides the "human" semantic primes,
//! achieving **Artificial Wisdom** requires grounding in **Ontological Primes**--
//! the irreducible atoms of mathematics, physics, geometry, and strategy.
//!
//! ## The Nine-Tier Primitive Hierarchy
//!
//! | Tier | Domain | Module |
//! |------|--------|--------|
//! | 0 | NSM Foundation (65 Wierzbicka primes) | `init_tiers` |
//! | 1 | Mathematical & Logical | `init_tiers` |
//! | 2 | Physical Reality | `init_tiers` |
//! | 3 | Geometric & Topological | `init_tiers` |
//! | 4 | Strategic & Social | `init_tiers` |
//! | 5 | Meta-Cognitive & Metabolic | `init_tiers` |
//! | 6 | Temporal (Allen's Interval Algebra) | `init_advanced_tiers` |
//! | 7 | Compositional operators | `init_advanced_tiers` |
//! | 8 | Consciousness-specific | `init_advanced_tiers` |
//! | 9 | Code & Symbol Manipulation | `init_advanced_tiers` |
//!
//! Additional domain-specific primitives (biological, emotional, ecological,
//! quantum, economic, linguistic, social/moral) are in `init_domains`.
//!
//! Derived primitives (uncertainty, physics extensions, information theory,
//! consciousness measurement) are in `init_derived`.
//!
//! ## Consciousness-Guided Validation
//!
//! Unlike traditional AI that assumes primitives help, this system uses the
//! **Consciousness Observatory** to empirically measure Phi improvements from
//! primitive-based reasoning. Primitives are validated, not assumed.
//!
//! ## Architecture: Domain Manifolds
//!
//! To maintain orthogonality with 250+ primitives in 16K-dimensional space,
//! we use **hierarchical binding**:
//!
//! ```rust,ignore
//! // Each domain gets a rotation in BinaryHV space
//! MATH_MANIFOLD = random_hv16();
//! ZERO = MATH_MANIFOLD ^ ZERO_LOCAL;
//! ONE = MATH_MANIFOLD ^ ONE_LOCAL;
//!
//! // This preserves orthogonality within and across domains
//! ```

// === Type definitions ===
mod primitive;
mod primitive_tier;

// === Data structures ===
mod composition_algebra;
mod composition_cache;
mod lsh_index;
mod persistence;
mod primitive_graph;

// === Initialization (impl PrimitiveSystem) ===
mod init_advanced_tiers;
mod init_derived;
mod init_domains;
mod init_tiers;

// === Public API (impl PrimitiveSystem) ===
mod operations;

#[cfg(test)]
mod tests;

// Re-exports: all public types remain accessible at this module level
pub use composition_algebra::*;
pub use composition_cache::*;
pub use lsh_index::*;
pub use persistence::*;
pub use primitive::*;
pub use primitive_graph::*;
pub use primitive_tier::*;

use crate::hdc::binary_hv::BinaryHV;
use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Global cached instance of PrimitiveSystem.
///
/// # Lazy Initialization
///
/// The system uses `once_cell::sync::Lazy` for optimal deferred initialization:
/// - **Zero startup cost**: No primitives are created until first access
/// - **Single initialization**: Built exactly once, then cached forever
/// - **Thread-safe**: Safe for concurrent access from multiple threads
///
/// # Memory Usage
///
/// When initialized, the system contains ~200 primitives across 9 tiers.
/// Each primitive stores a 16,384-bit BinaryHV encoding (~2KB), plus metadata.
/// Total memory: ~500KB for the complete ontological primitive system.
///
/// # Design Rationale
///
/// Per-primitive lazy initialization was considered but rejected because:
/// 1. Most use cases access multiple primitives for reasoning/composition
/// 2. Derived primitives depend on base primitives (complex dependency graph)
/// 3. System-level lazy already provides zero startup cost
/// 4. Added complexity would outweigh marginal memory savings
static GLOBAL_PRIMITIVE_SYSTEM: Lazy<PrimitiveSystem> = Lazy::new(PrimitiveSystem::new);

/// Generate a deterministic seed from a string name.
///
/// Uses FNV-1a (64-bit) which is stable across Rust versions and platforms,
/// unlike `DefaultHasher` whose algorithm is explicitly not guaranteed stable.
/// This ensures primitives always get the same encoding across runs,
/// compiler versions, and target architectures.
pub fn seed_from_name(name: &str) -> u64 {
    // FNV-1a 64-bit: well-known, stable, no dependencies
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in name.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The Primitive System - manages all ontological primes.
///
/// Contains primitives organized by tier and domain, with binding grammar
/// rules for composing primitives across tiers.
///
/// # Module Layout
///
/// - Construction: `new()`, `global()`, `derive_encoding()` (this file)
/// - Tier 0-5 init: `init_tiers`
/// - Domain init (bio, emotion, ecology, etc.): `init_domains`
/// - Tier 6-9 init: `init_advanced_tiers`
/// - Derived primitive init: `init_derived`
/// - Public query/search/composition API: `operations`
#[derive(Debug)]
pub struct PrimitiveSystem {
    /// All domain manifolds
    domains: HashMap<String, DomainManifold>,

    /// All primitives by name
    primitives: HashMap<String, Primitive>,

    /// Primitives organized by tier
    by_tier: HashMap<PrimitiveTier, Vec<String>>,

    /// Binding grammar rules
    binding_rules: Vec<BindingRule>,
}

impl PrimitiveSystem {
    /// Get a reference to the global cached PrimitiveSystem instance.
    ///
    /// This is the preferred way to access the PrimitiveSystem for read-only operations.
    /// The system is lazily initialized on first access and cached for subsequent calls.
    ///
    /// # Performance
    /// - First call: O(n) where n is the number of primitives to initialize
    /// - Subsequent calls: O(1) (returns cached reference)
    ///
    /// # Example
    /// ```rust,ignore
    /// let system = PrimitiveSystem::global();
    /// let zero = system.get("ZERO").unwrap();
    /// ```
    pub fn global() -> &'static PrimitiveSystem {
        &GLOBAL_PRIMITIVE_SYSTEM
    }

    /// Create new primitive system with all tiers initialized.
    pub fn new() -> Self {
        let mut system = Self {
            domains: HashMap::new(),
            primitives: HashMap::new(),
            by_tier: HashMap::new(),
            binding_rules: Vec::new(),
        };

        // Initialize all tiers
        // Tier 0: NSM (Natural Semantic Metalanguage) - 65 Wierzbicka primes
        system.init_tier0_nsm();

        system.init_tier1_mathematical();
        system.init_tier2_physical();
        system.init_tier3_geometric();
        system.init_tier4_strategic();
        system.init_tier5_metacognitive();

        // Initialize gap analysis additions (comprehensive ontology)
        // These add domain-specific primitives that may be referenced by derivations
        system.init_biological_primitives();
        system.init_emotional_primitives();
        system.init_ecological_primitives();
        system.init_quantum_primitives();
        system.init_economic_primitives();
        system.init_linguistic_primitives();
        system.init_social_moral_primitives();
        system.init_institutional_primitives();

        // Initialize Tier 6: Temporal primitives (Allen's Interval Algebra extended)
        system.init_tier6_temporal();

        // Initialize Tier 7: Compositional primitives (composition operators)
        system.init_tier7_compositional();

        // Initialize Tier 8: Consciousness-specific primitives
        // Qualia, attention, memory operations, and agency
        // MUST come before init_derived_primitives so SALIENCE/SELECTION exist
        system.init_consciousness_primitives();

        // Initialize Tier 9: Code primitives
        // Enables consciousness-aware code understanding, generation, and transformation
        system.init_tier9_code();

        // Initialize derived primitives (uncertainty, physics extensions, information theory)
        // These reference primitives from all tiers, so call LAST
        system.init_derived_primitives();

        system
    }

    /// Derive an encoding by binding parent primitive encodings together.
    ///
    /// If all parents are found, the result is their sequential XOR binding
    /// embedded in the given domain. If any parent is missing, falls back to
    /// a deterministic random vector seeded from `name`.
    ///
    /// NOTE: In debug builds, missing parents are logged to help identify
    /// registration ordering issues. The fallback exists because the current
    /// initialization order has derived primitives (Tier 1) initialized before
    /// consciousness base primitives (Tier 9) they may reference.
    fn derive_encoding(&self, name: &str, parents: &[&str], domain: &DomainManifold) -> BinaryHV {
        if parents.is_empty() {
            return domain.embed(BinaryHV::random(seed_from_name(name)));
        }

        let mut parent_encodings: Vec<&BinaryHV> = Vec::new();
        for parent_name in parents {
            match self.primitives.get(*parent_name) {
                Some(p) => parent_encodings.push(&p.encoding),
                None => {
                    // Parent not yet registered -- fall back to seeded random.
                    // This can happen when derived primitives reference parents
                    // from higher tiers that aren't initialized yet.
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "derive_encoding: '{}' parent '{}' not found (primitives count: {}), using seeded fallback",
                        name,
                        parent_name,
                        self.primitives.len()
                    );
                    return domain.embed(BinaryHV::random(seed_from_name(name)));
                }
            }
        }
        // Bind all parents sequentially.
        // NOTE: We do NOT re-embed in the domain because parent encodings are
        // already embedded in their respective domains. Re-embedding would add
        // an extra rotation that breaks the algebraic relationship:
        //   derived ^ parent1 should recover parent2
        // If we embedded, we'd get:
        //   domain.rotation ^ (parent1 ^ parent2) ^ parent1 = domain.rotation ^ parent2 != parent2
        let mut result = *parent_encodings[0];
        for enc in &parent_encodings[1..] {
            result = result.bind(enc);
        }
        result
    }
}
