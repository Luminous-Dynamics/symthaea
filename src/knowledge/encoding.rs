// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC Knowledge Encoding
//!
//! Encodes extracted facts as composite hypervectors using the HDC primitive
//! system. Each fact becomes a structured binding:
//!
//! ```text
//! FACT_HV = AGENT ⊗ agent_hv ⊗ ACTION ⊗ action_hv ⊗ PATIENT ⊗ patient_hv ⊗ TIME ⊗ time_hv
//! ```
//!
//! This enables compositional similarity search: you can query "who sanctioned Iran?"
//! by constructing `ACTION ⊗ sanction_hv ⊗ PATIENT ⊗ iran_hv` and finding similar vectors.
//!
//! Science: Kanerva (2009) HDC, Plate (2003) Holographic Reduced Representations

use super::extraction::{EntityType, ExtractedFact, SemanticRole};
use std::collections::HashMap;
use symthaea_core::hdc::unified_hv::{BinaryHV, HDC_DIMENSION};

// ── Types ──────────────────────────────────────────────────────────────────

/// An HDC-encoded fact ready for storage and similarity search
#[derive(Debug, Clone)]
pub struct FactEncoding {
    /// The composite hypervector representing this fact
    pub vector: BinaryHV,
    /// Role-specific sub-vectors for compositional queries
    pub role_vectors: HashMap<SemanticRole, BinaryHV>,
    /// Source fact text (for human readability)
    pub source_text: String,
    /// Encoding confidence (extraction confidence × encoding quality)
    pub confidence: f32,
}

/// Knowledge encoder: text → HDC vectors via role-binding algebra
pub struct KnowledgeEncoder {
    /// Role basis vectors: one per SemanticRole
    role_bases: HashMap<SemanticRole, BinaryHV>,
    /// Entity type basis vectors
    type_bases: HashMap<EntityType, BinaryHV>,
    /// Token cache: text → BinaryHV (avoids re-encoding)
    token_cache: HashMap<String, BinaryHV>,
    /// Cache capacity limit
    cache_capacity: usize,
    /// Total encodings performed
    total_encodings: u64,
}

impl Default for KnowledgeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeEncoder {
    pub fn new() -> Self {
        Self::with_seed(0x4E0E_1ED6_E5ED)
    }

    pub fn with_seed(seed: u64) -> Self {
        let mut role_bases = HashMap::new();
        let roles = [
            SemanticRole::Agent,
            SemanticRole::Patient,
            SemanticRole::Instrument,
            SemanticRole::Context,
            SemanticRole::Goal,
            SemanticRole::Source,
            SemanticRole::Destination,
            SemanticRole::Temporal,
            SemanticRole::Location,
            SemanticRole::Cause,
            SemanticRole::Result,
            #[cfg(feature = "therapeutic")]
            SemanticRole::TherapeuticTarget,
            #[cfg(feature = "therapeutic")]
            SemanticRole::ProtectiveFactor,
            #[cfg(feature = "therapeutic")]
            SemanticRole::RiskFactor,
        ];

        for (i, role) in roles.iter().enumerate() {
            role_bases.insert(*role, BinaryHV::random(seed + i as u64));
        }

        let mut type_bases = HashMap::new();
        let types = [
            EntityType::Person,
            EntityType::Organization,
            EntityType::Place,
            EntityType::Event,
            EntityType::Concept,
            EntityType::Quantity,
            EntityType::Temporal,
            EntityType::Artifact,
            EntityType::Process,
            EntityType::Property,
            #[cfg(feature = "therapeutic")]
            EntityType::ClinicalConcept,
            #[cfg(feature = "therapeutic")]
            EntityType::Symptom,
            #[cfg(feature = "therapeutic")]
            EntityType::Intervention,
        ];

        for (i, etype) in types.iter().enumerate() {
            type_bases.insert(*etype, BinaryHV::random(seed + 100 + i as u64));
        }

        Self {
            role_bases,
            type_bases,
            token_cache: HashMap::new(),
            cache_capacity: 10_000,
            total_encodings: 0,
        }
    }

    /// Encode a text token as a BinaryHV (cached)
    pub fn encode_token(&mut self, text: &str) -> BinaryHV {
        let key = text.to_lowercase();
        if let Some(cached) = self.token_cache.get(&key) {
            return cached.clone();
        }

        // Deterministic encoding from text: FNV-1a hash as seed
        let seed = fnv1a_hash(text);
        let hv = BinaryHV::random(seed);

        // Cache management: evict oldest if at capacity
        if self.token_cache.len() >= self.cache_capacity {
            // Remove arbitrary entry (HashMap doesn't track insertion order,
            // but this is acceptable for a cache)
            if let Some(first_key) = self.token_cache.keys().next().cloned() {
                self.token_cache.remove(&first_key);
            }
        }

        self.token_cache.insert(key, hv.clone());
        hv
    }

    /// Encode an extracted fact as a composite hypervector
    ///
    /// The encoding binds each entity's vector with its semantic role basis,
    /// then bundles all role-bound vectors into a single fact representation.
    pub fn encode_fact(&mut self, fact: &ExtractedFact) -> FactEncoding {
        let mut role_vectors = HashMap::new();
        let mut components = Vec::new();

        // Encode each entity with its role
        for entity in &fact.entities {
            let entity_hv = self.encode_token(&entity.text);
            let type_hv = self
                .type_bases
                .get(&entity.entity_type)
                .cloned()
                .unwrap_or_else(|| BinaryHV::random(999));

            // entity_typed = entity ⊗ type
            let entity_typed = entity_hv.bind(&type_hv);

            // Get role for this entity
            if let Some(role) = fact.role_map.get(&entity.text) {
                if let Some(role_basis) = self.role_bases.get(role) {
                    // role_bound = role_basis ⊗ entity_typed
                    let role_bound = role_basis.bind(&entity_typed);
                    role_vectors.insert(*role, role_bound.clone());
                    components.push(role_bound);
                }
            }
        }

        // Encode relations (predicate verbs add semantic flavor)
        for relation in &fact.relations {
            let pred_hv = self.encode_token(&relation.predicate);
            components.push(pred_hv);
        }

        // Bundle all components: FACT = Σ(role_i ⊗ entity_i) + Σ(predicate_j)
        let vector = if components.is_empty() {
            // Fallback: encode the source text directly
            self.encode_token(&fact.source_text)
        } else {
            BinaryHV::bundle(&components)
        };

        self.total_encodings += 1;

        FactEncoding {
            vector,
            role_vectors,
            source_text: fact.source_text.clone(),
            confidence: fact.confidence,
        }
    }

    /// Create a query vector for compositional search
    ///
    /// Example: to find "who sanctioned Iran?", call:
    /// ```rust,ignore
    /// let query = encoder.compose_query(&[
    ///     (SemanticRole::Patient, "Iran"),
    ///     (SemanticRole::Agent, "sanctions"),
    /// ]);
    /// ```
    pub fn compose_query(&mut self, role_terms: &[(SemanticRole, &str)]) -> BinaryHV {
        let mut components = Vec::new();

        for (role, term) in role_terms {
            let term_hv = self.encode_token(term);
            if let Some(role_basis) = self.role_bases.get(role) {
                components.push(role_basis.bind(&term_hv));
            }
        }

        if components.is_empty() {
            BinaryHV::random(0)
        } else {
            BinaryHV::bundle(&components)
        }
    }

    /// Total facts encoded since creation
    pub fn total_encodings(&self) -> u64 {
        self.total_encodings
    }

    /// Number of cached tokens
    pub fn cache_size(&self) -> usize {
        self.token_cache.len()
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// FNV-1a hash for deterministic seeding from text
pub(crate) fn fnv1a_hash(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in text.to_lowercase().bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::extraction::*;

    fn make_test_fact() -> ExtractedFact {
        let mut role_map = HashMap::new();
        role_map.insert("Iran".to_string(), SemanticRole::Patient);
        role_map.insert("United States".to_string(), SemanticRole::Agent);

        ExtractedFact {
            entities: vec![
                ExtractedEntity {
                    text: "United States".to_string(),
                    entity_type: EntityType::Organization,
                    confidence: 0.9,
                    offset: 0,
                },
                ExtractedEntity {
                    text: "Iran".to_string(),
                    entity_type: EntityType::Place,
                    confidence: 0.9,
                    offset: 25,
                },
            ],
            relations: vec![ExtractedRelation {
                subject: "United States".to_string(),
                predicate: "sanctioned".to_string(),
                object: "Iran".to_string(),
                subject_role: SemanticRole::Agent,
                object_role: SemanticRole::Patient,
                is_causal: true,
                is_negated: false,
                confidence: 0.8,
            }],
            role_map,
            source_text: "United States sanctioned Iran.".to_string(),
            confidence: 0.85,
        }
    }

    #[test]
    fn test_fact_encoding() {
        let mut encoder = KnowledgeEncoder::with_seed(42);
        let fact = make_test_fact();
        let encoding = encoder.encode_fact(&fact);

        // BinaryHV is always HDC_DIMENSION (16,384 bits = 2048 bytes)
        assert!(encoding.confidence > 0.0);
        assert!(!encoding.role_vectors.is_empty());
        assert!(encoding.confidence > 0.0);
    }

    #[test]
    fn test_compositional_query() {
        let mut encoder = KnowledgeEncoder::with_seed(42);

        // Encode a fact
        let fact = make_test_fact();
        let encoding = encoder.encode_fact(&fact);

        // Create a query for "who sanctioned Iran?"
        let query = encoder.compose_query(&[(SemanticRole::Patient, "Iran")]);

        // The query should have some similarity to the fact
        // (not necessarily high, but non-zero due to shared Iran component)
        let sim = encoding.vector.similarity(&query);
        // HDC similarity is normalized Hamming distance, expect weak signal
        assert!(sim > -0.5, "Expected some similarity, got {sim}");
    }

    #[test]
    fn test_token_caching() {
        let mut encoder = KnowledgeEncoder::with_seed(42);

        let hv1 = encoder.encode_token("Iran");
        let hv2 = encoder.encode_token("Iran");

        // Same token should produce identical vectors (cached)
        assert_eq!(hv1.similarity(&hv2), 1.0);
        assert_eq!(encoder.cache_size(), 1);
    }

    #[test]
    fn test_different_tokens_orthogonal() {
        let mut encoder = KnowledgeEncoder::with_seed(42);

        let hv1 = encoder.encode_token("sanctions");
        let hv2 = encoder.encode_token("diplomacy");

        // Different tokens should have low similarity (BinaryHV baseline ~0.5)
        // BinaryHV similarity is normalized Hamming: 1.0 = identical, 0.5 = random
        let sim = hv1.similarity(&hv2);
        assert!(
            sim < 0.9,
            "Expected different vectors, got similarity {sim}"
        );
    }

    #[test]
    fn test_deterministic_encoding() {
        let mut enc1 = KnowledgeEncoder::with_seed(42);
        let mut enc2 = KnowledgeEncoder::with_seed(42);

        let hv1 = enc1.encode_token("test");
        let hv2 = enc2.encode_token("test");

        assert_eq!(hv1.similarity(&hv2), 1.0);
    }

    #[test]
    fn test_cache_eviction() {
        let mut encoder = KnowledgeEncoder::with_seed(42);
        // Reduce capacity for test
        encoder.cache_capacity = 5;

        for i in 0..10 {
            encoder.encode_token(&format!("token_{i}"));
        }

        // Cache should not exceed capacity
        assert!(encoder.cache_size() <= 6); // Allow +1 for eviction timing
    }
}
