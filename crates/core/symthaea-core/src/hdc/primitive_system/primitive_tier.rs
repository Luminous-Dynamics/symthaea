// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::seed_from_name;
use crate::hdc::binary_hv::BinaryHV;
use serde::{Deserialize, Serialize};

/// Primitive tier in the hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum PrimitiveTier {
    /// Tier 0: NSM (implemented in vocabulary.rs)
    #[default]
    NSM,
    /// Tier 1: Mathematical & Logical
    Mathematical,
    /// Tier 2: Physical Reality
    Physical,
    /// Tier 3: Geometric & Topological
    Geometric,
    /// Tier 4: Strategic & Social
    Strategic,
    /// Tier 5: Meta-Cognitive & Metabolic
    MetaCognitive,
    /// Tier 6: Temporal Primitives (Allen's Interval Algebra)
    /// Enables reasoning about temporal relationships between consciousness states
    Temporal,
    /// Tier 7: Compositionality Primitives
    /// PARADIGM SHIFT: Complete algebra for combining primitives into higher-order structures!
    /// - Sequential (∘), Parallel (||), Conditional (?), Fixed-point (μ), Higher-order (↑)
    ///   Enables infinite complexity from finite base primitives through composition!
    Compositional,
    /// Tier 8: Consciousness-Specific Primitives
    /// First-person phenomenal experience, attention, memory operations, and agency.
    /// - QUALE, PHENOMENAL_BINDING, ATTEND, SALIENCE, REMEMBER, CONSOLIDATE, INTEND
    ///   Enables reasoning about subjective experience and conscious states!
    Consciousness,
    /// Tier 9: Code & Symbol Manipulation Primitives
    /// Enables consciousness-aware code understanding, generation, and transformation.
    /// - PARSE, ENTITY, ENCODE, GENERATE, COMPOSE, BRANCH, LOOP, DEBUG, VERIFY
    ///   Code operations flow through the same primitive routing as all other cognitive tasks!
    Code,
}

/// Domain manifold - a rotation in BinaryHV space for domain isolation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainManifold {
    /// Name of this domain
    pub name: String,
    /// Tier in hierarchy
    pub tier: PrimitiveTier,
    /// Rotation vector for this domain
    pub rotation: BinaryHV,
    /// Description of domain's purpose
    pub purpose: String,
}

impl DomainManifold {
    /// Create a new domain with deterministic rotation based on name
    pub fn new(name: impl Into<String>, tier: PrimitiveTier, purpose: impl Into<String>) -> Self {
        let name_str = name.into();
        let seed = seed_from_name(&name_str);
        Self {
            name: name_str,
            tier,
            rotation: BinaryHV::random(seed),
            purpose: purpose.into(),
        }
    }

    /// Embed a local primitive vector into this domain's manifold.
    ///
    /// Uses bind (XOR) rather than bundle so that:
    /// - The embedding is invertible: `embedded.bind(&rotation) == local_vector`
    /// - Hamming distances are preserved within each domain
    /// - Cross-domain orthogonality follows from random rotation vectors
    pub fn embed(&self, local_vector: BinaryHV) -> BinaryHV {
        self.rotation.bind(&local_vector)
    }

    /// Extract the local vector from a domain-embedded vector.
    ///
    /// Since embed uses XOR (self-inverse), unbinding is the same operation:
    /// `unembed(embed(v)) == v`
    pub fn unembed(&self, embedded: &BinaryHV) -> BinaryHV {
        self.rotation.bind(embedded)
    }
}
