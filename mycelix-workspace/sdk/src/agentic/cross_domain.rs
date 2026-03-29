// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cross-Domain Trust Translation
//!
//! Translate K-Vector trust profiles between domains with different trust semantics.
//!
//! ## Philosophy
//!
//! Trust is contextual. An agent trusted in one domain (e.g., code review) may need
//! different trust characteristics in another domain (e.g., financial operations).
//! This module provides:
//!
//! - Domain definitions with trust dimension relevance
//! - Translation functions that map trust across domains
//! - Confidence factors for translation quality
//! - Verifiable translation via ZK proofs
//!
//! ## Key Concepts
//!
//! - **TrustDomain**: A context with specific trust requirements
//! - **DomainRelevance**: How relevant each K-Vector dimension is to a domain
//! - **TranslationPath**: A sequence of domains for multi-hop translation
//! - **TranslationConfidence**: How reliable the translation is

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::matl::{KVector, KVectorDimension, KVECTOR_WEIGHTS_ARRAY};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Trust Domains
// ============================================================================

/// A context/domain with specific trust requirements
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct TrustDomain {
    /// Unique identifier for this domain
    pub domain_id: String,

    /// Human-readable name
    pub name: String,

    /// Description of the domain's trust requirements
    pub description: String,

    /// How relevant each K-Vector dimension is to this domain (0.0-1.0)
    pub dimension_relevance: DomainRelevance,

    /// Minimum trust score required for participation
    pub min_trust_threshold: f32,

    /// Whether this domain requires verification (k_v)
    pub requires_verification: bool,

    /// Domain-specific trust weight adjustments
    pub weight_overrides: Option<HashMap<String, f32>>,
}

/// Relevance of each K-Vector dimension to a domain
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct DomainRelevance {
    /// How relevant reputation is to this domain
    pub reputation: f32,
    /// How relevant activity is
    pub activity: f32,
    /// How relevant integrity is
    pub integrity: f32,
    /// How relevant performance is
    pub performance: f32,
    /// How relevant membership duration is
    pub membership: f32,
    /// How relevant economic stake is
    pub stake: f32,
    /// How relevant historical consistency is
    pub historical: f32,
    /// How relevant network topology is
    pub topology: f32,
    /// How relevant identity verification is
    pub verification: f32,
    /// How relevant coherence (Phi) is
    pub coherence: f32,
}

impl DomainRelevance {
    /// Create a balanced relevance (all dimensions equally relevant)
    pub fn balanced() -> Self {
        Self {
            reputation: 1.0,
            activity: 1.0,
            integrity: 1.0,
            performance: 1.0,
            membership: 1.0,
            stake: 1.0,
            historical: 1.0,
            topology: 1.0,
            verification: 1.0,
            coherence: 1.0,
        }
    }

    /// Convert to array for computation
    pub fn to_array(&self) -> [f32; 10] {
        [
            self.reputation,
            self.activity,
            self.integrity,
            self.performance,
            self.membership,
            self.stake,
            self.historical,
            self.topology,
            self.verification,
            self.coherence,
        ]
    }

    /// Get relevance for a specific dimension
    pub fn get(&self, dim: KVectorDimension) -> f32 {
        match dim {
            KVectorDimension::Reputation => self.reputation,
            KVectorDimension::Activity => self.activity,
            KVectorDimension::Integrity => self.integrity,
            KVectorDimension::Performance => self.performance,
            KVectorDimension::Membership => self.membership,
            KVectorDimension::Stake => self.stake,
            KVectorDimension::Historical => self.historical,
            KVectorDimension::Topology => self.topology,
            KVectorDimension::Verification => self.verification,
            KVectorDimension::Coherence => self.coherence,
        }
    }

    /// Compute similarity between two domain relevance profiles
    pub fn similarity(&self, other: &DomainRelevance) -> f32 {
        let a = self.to_array();
        let b = other.to_array();

        // Cosine similarity
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a > 0.0 && mag_b > 0.0 {
            dot / (mag_a * mag_b)
        } else {
            0.0
        }
    }
}

impl Default for DomainRelevance {
    fn default() -> Self {
        Self::balanced()
    }
}

// ============================================================================
// Predefined Domains
// ============================================================================

/// Predefined trust domain templates
pub struct DomainTemplates;

impl DomainTemplates {
    /// Financial operations domain - high stake and integrity requirements
    pub fn financial() -> TrustDomain {
        TrustDomain {
            domain_id: "financial".to_string(),
            name: "Financial Operations".to_string(),
            description: "High-stakes financial transactions requiring verified identity and stake"
                .to_string(),
            dimension_relevance: DomainRelevance {
                reputation: 0.8,
                activity: 0.3,  // Activity less important
                integrity: 1.0, // Critical
                performance: 0.5,
                membership: 0.7,   // Longevity matters
                stake: 1.0,        // Critical
                historical: 0.9,   // Consistency important
                topology: 0.2,     // Less relevant
                verification: 1.0, // Critical
                coherence: 0.6,    // Moderate importance for consistency
            },
            min_trust_threshold: 0.6,
            requires_verification: true,
            weight_overrides: None,
        }
    }

    /// Code review domain - high performance and integrity
    pub fn code_review() -> TrustDomain {
        TrustDomain {
            domain_id: "code_review".to_string(),
            name: "Code Review".to_string(),
            description: "Technical review of code changes".to_string(),
            dimension_relevance: DomainRelevance {
                reputation: 0.8,
                activity: 0.6,     // Active reviewers preferred
                integrity: 0.9,    // Important for honest reviews
                performance: 1.0,  // Critical - quality of reviews
                membership: 0.4,   // Less important
                stake: 0.2,        // Not very relevant
                historical: 0.7,   // Consistent review quality
                topology: 0.5,     // Some connectivity helps
                verification: 0.5, // Moderate
                coherence: 0.8,    // High - coherent thinking needed
            },
            min_trust_threshold: 0.4,
            requires_verification: false,
            weight_overrides: None,
        }
    }

    /// Social/community domain - reputation and activity focused
    pub fn social() -> TrustDomain {
        TrustDomain {
            domain_id: "social".to_string(),
            name: "Social Community".to_string(),
            description: "Community interactions and social activities".to_string(),
            dimension_relevance: DomainRelevance {
                reputation: 1.0,   // Critical
                activity: 0.9,     // Very important
                integrity: 0.6,    // Moderate
                performance: 0.5,  // Moderate
                membership: 0.6,   // Some importance
                stake: 0.1,        // Not very relevant
                historical: 0.5,   // Moderate
                topology: 1.0,     // Critical - network position
                verification: 0.3, // Less important
                coherence: 0.4,    // Low - social interactions varied
            },
            min_trust_threshold: 0.3,
            requires_verification: false,
            weight_overrides: None,
        }
    }

    /// Governance domain - balanced with emphasis on stake and reputation
    pub fn governance() -> TrustDomain {
        TrustDomain {
            domain_id: "governance".to_string(),
            name: "Governance".to_string(),
            description: "Participation in governance decisions".to_string(),
            dimension_relevance: DomainRelevance {
                reputation: 1.0,   // Critical
                activity: 0.5,     // Some importance
                integrity: 0.9,    // Very important
                performance: 0.6,  // Moderate
                membership: 0.8,   // Longevity matters
                stake: 0.9,        // Very important
                historical: 0.8,   // Important
                topology: 0.6,     // Moderate
                verification: 0.7, // Important
                coherence: 0.7,    // High - rational decision making
            },
            min_trust_threshold: 0.5,
            requires_verification: false,
            weight_overrides: None,
        }
    }

    /// Research/academic domain - performance and reputation focused
    pub fn research() -> TrustDomain {
        TrustDomain {
            domain_id: "research".to_string(),
            name: "Research".to_string(),
            description: "Academic and research contributions".to_string(),
            dimension_relevance: DomainRelevance {
                reputation: 0.9,
                activity: 0.4,
                integrity: 0.8,
                performance: 1.0, // Critical
                membership: 0.3,
                stake: 0.2,
                historical: 0.7,
                topology: 0.6, // Collaboration networks matter
                verification: 0.6,
                coherence: 0.9, // Very high - research requires coherent thinking
            },
            min_trust_threshold: 0.4,
            requires_verification: false,
            weight_overrides: None,
        }
    }

    /// Infrastructure/operations domain - high integrity and verification
    pub fn infrastructure() -> TrustDomain {
        TrustDomain {
            domain_id: "infrastructure".to_string(),
            name: "Infrastructure".to_string(),
            description: "System operations and infrastructure management".to_string(),
            dimension_relevance: DomainRelevance {
                reputation: 0.7,
                activity: 0.5,
                integrity: 1.0,   // Critical
                performance: 0.9, // Very important
                membership: 0.6,
                stake: 0.5,
                historical: 0.9, // Consistency critical
                topology: 0.4,
                verification: 1.0, // Critical
                coherence: 0.8,    // High - operational consistency
            },
            min_trust_threshold: 0.6,
            requires_verification: true,
            weight_overrides: None,
        }
    }

    /// Get all predefined domains
    pub fn all() -> Vec<TrustDomain> {
        vec![
            Self::financial(),
            Self::code_review(),
            Self::social(),
            Self::governance(),
            Self::research(),
            Self::infrastructure(),
        ]
    }
}

// ============================================================================
// Trust Translation
// ============================================================================

/// Result of translating trust between domains
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct TranslationResult {
    /// Original K-Vector
    pub source_kvector: KVector,

    /// Translated K-Vector for target domain
    pub target_kvector: KVector,

    /// Source domain
    pub source_domain: String,

    /// Target domain
    pub target_domain: String,

    /// Trust score in source domain
    pub source_trust: f32,

    /// Trust score in target domain
    pub target_trust: f32,

    /// Confidence in the translation (0.0-1.0)
    pub translation_confidence: f32,

    /// Breakdown of how each dimension translated
    pub dimension_translations: Vec<DimensionTranslation>,

    /// Whether the agent meets target domain requirements
    pub meets_target_requirements: bool,

    /// Warnings about the translation
    pub warnings: Vec<String>,
}

/// How a single dimension was translated
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct DimensionTranslation {
    /// The dimension
    pub dimension: KVectorDimension,

    /// Source value
    pub source_value: f32,

    /// Target value (after translation)
    pub target_value: f32,

    /// Source domain relevance
    pub source_relevance: f32,

    /// Target domain relevance
    pub target_relevance: f32,

    /// Translation factor applied
    pub translation_factor: f32,
}

/// Translate a K-Vector between domains
pub fn translate_trust(
    kvector: &KVector,
    source_domain: &TrustDomain,
    target_domain: &TrustDomain,
) -> TranslationResult {
    let source_arr = kvector.to_array();
    let source_rel = source_domain.dimension_relevance.to_array();
    let target_rel = target_domain.dimension_relevance.to_array();

    // Compute domain similarity for base confidence
    let domain_similarity = source_domain
        .dimension_relevance
        .similarity(&target_domain.dimension_relevance);

    // Translate each dimension (10 dimensions including k_coherence)
    let mut target_arr = [0.0f32; 10];
    let mut dimension_translations = Vec::with_capacity(10);
    let mut total_transfer = 0.0f32;
    let mut total_weight = 0.0f32;

    for (i, dim) in KVectorDimension::all().iter().enumerate() {
        let source_val = source_arr[i];
        let source_relevance = source_rel[i];
        let target_relevance = target_rel[i];

        // Translation factor based on relevance ratio
        // If target cares more about a dimension than source, we're conservative
        // If target cares less, we can preserve more trust
        let translation_factor = if source_relevance > 0.0 {
            (target_relevance / source_relevance).min(1.0) * domain_similarity.sqrt()
        } else {
            0.0 // Can't transfer from irrelevant source dimension
        };

        // Apply translation with damping for low-relevance dimensions
        let target_val = source_val * translation_factor;
        target_arr[i] = target_val.clamp(0.0, 1.0);

        // Track transfer quality
        let weight = target_relevance * KVECTOR_WEIGHTS_ARRAY[i];
        total_transfer += translation_factor * weight;
        total_weight += weight;

        dimension_translations.push(DimensionTranslation {
            dimension: *dim,
            source_value: source_val,
            target_value: target_arr[i],
            source_relevance,
            target_relevance,
            translation_factor,
        });
    }

    let target_kvector = KVector::from_array(target_arr);

    // Compute trust scores
    let source_trust = compute_domain_trust(kvector, source_domain);
    let target_trust = compute_domain_trust(&target_kvector, target_domain);

    // Translation confidence based on domain similarity and transfer quality
    let transfer_quality = if total_weight > 0.0 {
        total_transfer / total_weight
    } else {
        0.0
    };
    let translation_confidence = (domain_similarity * 0.4 + transfer_quality * 0.6).clamp(0.0, 1.0);

    // Check requirements
    let meets_threshold = target_trust >= target_domain.min_trust_threshold;
    let meets_verification = !target_domain.requires_verification || kvector.is_verified();
    let meets_target_requirements = meets_threshold && meets_verification;

    // Generate warnings
    let mut warnings = Vec::new();
    if translation_confidence < 0.5 {
        warnings.push(format!(
            "Low translation confidence ({:.0}%): domains have different trust priorities",
            translation_confidence * 100.0
        ));
    }
    if target_trust < source_trust * 0.5 {
        warnings.push(format!(
            "Significant trust reduction: {:.0}% of source trust preserved",
            (target_trust / source_trust) * 100.0
        ));
    }
    if target_domain.requires_verification && !kvector.is_verified() {
        warnings.push("Target domain requires verification but agent is not verified".to_string());
    }

    TranslationResult {
        source_kvector: *kvector,
        target_kvector,
        source_domain: source_domain.domain_id.clone(),
        target_domain: target_domain.domain_id.clone(),
        source_trust,
        target_trust,
        translation_confidence,
        dimension_translations,
        meets_target_requirements,
        warnings,
    }
}

/// Compute trust score weighted by domain relevance
pub fn compute_domain_trust(kvector: &KVector, domain: &TrustDomain) -> f32 {
    let arr = kvector.to_array();
    let rel = domain.dimension_relevance.to_array();

    // Weighted average with domain relevance
    let mut weighted_sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for i in 0..10 {
        let base_weight = KVECTOR_WEIGHTS_ARRAY[i];
        let domain_weight = base_weight * rel[i];
        weighted_sum += arr[i] * domain_weight;
        weight_sum += domain_weight;
    }

    if weight_sum > 0.0 {
        weighted_sum / weight_sum
    } else {
        0.0
    }
}

// ============================================================================
// Multi-Hop Translation
// ============================================================================

/// A path through multiple domains for trust translation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranslationPath {
    /// Sequence of domains (source → intermediate₁ → ... → target)
    pub domains: Vec<TrustDomain>,

    /// Results at each hop
    pub hop_results: Vec<TranslationResult>,

    /// Final K-Vector
    pub final_kvector: KVector,

    /// Cumulative confidence (product of hop confidences)
    pub cumulative_confidence: f32,

    /// Total trust degradation
    pub trust_degradation: f32,
}

/// Translate through a path of domains
pub fn translate_path(kvector: &KVector, domains: &[TrustDomain]) -> Option<TranslationPath> {
    if domains.len() < 2 {
        return None;
    }

    let mut current = *kvector;
    let mut hop_results = Vec::with_capacity(domains.len() - 1);
    let mut cumulative_confidence = 1.0f32;

    for window in domains.windows(2) {
        let source = &window[0];
        let target = &window[1];
        let result = translate_trust(&current, source, target);

        cumulative_confidence *= result.translation_confidence;
        current = result.target_kvector;
        hop_results.push(result);
    }

    let initial_trust = kvector.trust_score();
    let final_trust = current.trust_score();
    let trust_degradation = if initial_trust > 0.0 {
        1.0 - (final_trust / initial_trust)
    } else {
        1.0
    };

    Some(TranslationPath {
        domains: domains.to_vec(),
        hop_results,
        final_kvector: current,
        cumulative_confidence,
        trust_degradation,
    })
}

// ============================================================================
// Domain Registry
// ============================================================================

/// Registry of known trust domains
pub struct DomainRegistry {
    domains: HashMap<String, TrustDomain>,
}

impl DomainRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            domains: HashMap::new(),
        }
    }

    /// Create a registry with all predefined domains
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        for domain in DomainTemplates::all() {
            registry.register(domain);
        }
        registry
    }

    /// Register a new domain
    pub fn register(&mut self, domain: TrustDomain) {
        self.domains.insert(domain.domain_id.clone(), domain);
    }

    /// Get a domain by ID
    pub fn get(&self, domain_id: &str) -> Option<&TrustDomain> {
        self.domains.get(domain_id)
    }

    /// List all domain IDs
    pub fn list_domains(&self) -> Vec<String> {
        self.domains.keys().cloned().collect()
    }

    /// Translate between registered domains by ID
    pub fn translate(
        &self,
        kvector: &KVector,
        source_id: &str,
        target_id: &str,
    ) -> Option<TranslationResult> {
        let source = self.domains.get(source_id)?;
        let target = self.domains.get(target_id)?;
        Some(translate_trust(kvector, source, target))
    }

    /// Find the best path between two domains (minimizing trust degradation)
    pub fn find_best_path(
        &self,
        kvector: &KVector,
        source_id: &str,
        target_id: &str,
        max_hops: usize,
    ) -> Option<TranslationPath> {
        let source = self.domains.get(source_id)?;
        let target = self.domains.get(target_id)?;

        // Direct path
        let direct_result = translate_trust(kvector, source, target);
        let direct_path = TranslationPath {
            domains: vec![source.clone(), target.clone()],
            hop_results: vec![direct_result.clone()],
            final_kvector: direct_result.target_kvector,
            cumulative_confidence: direct_result.translation_confidence,
            trust_degradation: if kvector.trust_score() > 0.0 {
                1.0 - (direct_result.target_trust / kvector.trust_score())
            } else {
                1.0
            },
        };

        if max_hops <= 1 {
            return Some(direct_path);
        }

        // Try paths through intermediate domains
        let mut best_path = direct_path;

        for (id, intermediate) in &self.domains {
            if id == source_id || id == target_id {
                continue;
            }

            let path_domains = vec![source.clone(), intermediate.clone(), target.clone()];
            if let Some(path) = translate_path(kvector, &path_domains) {
                // Prefer paths with higher confidence and lower degradation
                let path_score = path.cumulative_confidence * (1.0 - path.trust_degradation);
                let best_score =
                    best_path.cumulative_confidence * (1.0 - best_path.trust_degradation);

                if path_score > best_score {
                    best_path = path;
                }
            }
        }

        Some(best_path)
    }

    /// Get domain similarity matrix
    pub fn similarity_matrix(&self) -> HashMap<(String, String), f32> {
        let mut matrix = HashMap::new();
        let ids: Vec<_> = self.domains.keys().cloned().collect();

        for i in 0..ids.len() {
            for j in 0..ids.len() {
                if let (Some(d1), Some(d2)) = (self.domains.get(&ids[i]), self.domains.get(&ids[j]))
                {
                    let similarity = d1.dimension_relevance.similarity(&d2.dimension_relevance);
                    matrix.insert((ids[i].clone(), ids[j].clone()), similarity);
                }
            }
        }

        matrix
    }
}

impl Default for DomainRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ============================================================================
// Domain Compatibility
// ============================================================================

/// Analyze compatibility between domains
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainCompatibility {
    /// Source domain
    pub source: String,
    /// Target domain
    pub target: String,
    /// Overall compatibility score (0.0-1.0)
    pub compatibility: f32,
    /// Which dimensions transfer well
    pub strong_transfers: Vec<KVectorDimension>,
    /// Which dimensions transfer poorly
    pub weak_transfers: Vec<KVectorDimension>,
    /// Recommended actions for better translation
    pub recommendations: Vec<String>,
}

/// Analyze compatibility between two domains
pub fn analyze_domain_compatibility(
    source: &TrustDomain,
    target: &TrustDomain,
) -> DomainCompatibility {
    let source_rel = source.dimension_relevance.to_array();
    let target_rel = target.dimension_relevance.to_array();

    let mut strong_transfers = Vec::new();
    let mut weak_transfers = Vec::new();
    let mut recommendations = Vec::new();

    for (i, dim) in KVectorDimension::all().iter().enumerate() {
        let transfer_quality = if source_rel[i] > 0.0 {
            (target_rel[i] / source_rel[i]).min(1.0)
        } else {
            0.0
        };

        if transfer_quality >= 0.7 && target_rel[i] >= 0.5 {
            strong_transfers.push(*dim);
        } else if transfer_quality < 0.3 && target_rel[i] >= 0.5 {
            weak_transfers.push(*dim);
        }
    }

    // Generate recommendations
    if target.requires_verification && !source.requires_verification {
        recommendations
            .push("Target domain requires verification - ensure agent is verified".to_string());
    }

    for dim in &weak_transfers {
        recommendations.push(format!(
            "Build {} in target domain context for better translation",
            dim.name()
        ));
    }

    if target.min_trust_threshold > source.min_trust_threshold {
        recommendations.push(format!(
            "Target has higher trust threshold ({:.0}% vs {:.0}%) - may need trust building",
            target.min_trust_threshold * 100.0,
            source.min_trust_threshold * 100.0
        ));
    }

    let compatibility = source
        .dimension_relevance
        .similarity(&target.dimension_relevance);

    DomainCompatibility {
        source: source.domain_id.clone(),
        target: target.domain_id.clone(),
        compatibility,
        strong_transfers,
        weak_transfers,
        recommendations,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_kvector() -> KVector {
        KVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.75, 0.7)
    }

    #[test]
    fn test_domain_relevance_similarity() {
        let balanced = DomainRelevance::balanced();
        let financial = DomainTemplates::financial().dimension_relevance;
        let social = DomainTemplates::social().dimension_relevance;

        // Balanced should be similar to itself
        assert!((balanced.similarity(&balanced) - 1.0).abs() < 0.001);

        // Financial and social should be less similar
        let fin_soc_sim = financial.similarity(&social);
        assert!(fin_soc_sim < 0.95); // Not identical
        assert!(fin_soc_sim > 0.3); // But not completely different
    }

    #[test]
    fn test_simple_translation() {
        let kv = test_kvector();
        let source = DomainTemplates::code_review();
        let target = DomainTemplates::financial();

        let result = translate_trust(&kv, &source, &target);

        // Should have lower trust in financial domain (more demanding)
        assert!(result.target_trust <= result.source_trust);

        // Confidence should be reasonable
        assert!(result.translation_confidence > 0.0);
        assert!(result.translation_confidence <= 1.0);

        // Should have dimension translations (10 dimensions now)
        assert_eq!(result.dimension_translations.len(), 10);
    }

    #[test]
    fn test_similar_domain_translation() {
        let kv = test_kvector();
        let domain1 = DomainTemplates::governance();
        let domain2 = DomainTemplates::governance(); // Same domain

        let result = translate_trust(&kv, &domain1, &domain2);

        // Same domain should have high confidence
        assert!((result.translation_confidence - 1.0).abs() < 0.01);

        // Trust should be preserved
        assert!((result.target_trust - result.source_trust).abs() < 0.01);
    }

    #[test]
    fn test_verification_requirement() {
        let unverified = KVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.3, 0.5); // k_v = 0.3
        let source = DomainTemplates::social();
        let target = DomainTemplates::financial(); // Requires verification

        let result = translate_trust(&unverified, &source, &target);

        // Should not meet requirements
        assert!(!result.meets_target_requirements);

        // Should have warning about verification
        assert!(result.warnings.iter().any(|w| w.contains("verification")));
    }

    #[test]
    fn test_domain_registry() {
        let registry = DomainRegistry::with_defaults();

        // Should have all predefined domains
        assert!(registry.get("financial").is_some());
        assert!(registry.get("code_review").is_some());
        assert!(registry.get("social").is_some());
        assert!(registry.get("governance").is_some());
        assert!(registry.get("research").is_some());
        assert!(registry.get("infrastructure").is_some());

        // Should be able to translate
        let kv = test_kvector();
        let result = registry.translate(&kv, "social", "governance").unwrap();
        assert!(result.translation_confidence > 0.0);
    }

    #[test]
    fn test_path_translation() {
        let kv = test_kvector();
        let domains = vec![
            DomainTemplates::social(),
            DomainTemplates::governance(),
            DomainTemplates::financial(),
        ];

        let path = translate_path(&kv, &domains).unwrap();

        // Should have 2 hops
        assert_eq!(path.hop_results.len(), 2);

        // Cumulative confidence should be product of hop confidences
        let expected_conf: f32 = path
            .hop_results
            .iter()
            .map(|r| r.translation_confidence)
            .product();
        assert!((path.cumulative_confidence - expected_conf).abs() < 0.001);
    }

    #[test]
    fn test_find_best_path() {
        let kv = test_kvector();
        let registry = DomainRegistry::with_defaults();

        let path = registry
            .find_best_path(&kv, "social", "financial", 2)
            .unwrap();

        // Should find a path
        assert!(!path.domains.is_empty());

        // Path should start at social and end at financial
        assert_eq!(path.domains.first().unwrap().domain_id, "social");
        assert_eq!(path.domains.last().unwrap().domain_id, "financial");
    }

    #[test]
    fn test_domain_compatibility_analysis() {
        let code_review = DomainTemplates::code_review();
        let research = DomainTemplates::research();

        let compat = analyze_domain_compatibility(&code_review, &research);

        // These domains should be reasonably compatible (both technical)
        assert!(compat.compatibility > 0.5);

        // Should have some strong transfers
        assert!(!compat.strong_transfers.is_empty());
    }

    #[test]
    fn test_dimension_translation_details() {
        let kv = test_kvector();
        let source = DomainTemplates::social(); // High reputation/topology relevance
        let target = DomainTemplates::financial(); // High stake/verification relevance

        let result = translate_trust(&kv, &source, &target);

        // Find integrity translation (should transfer well - important in both)
        let integrity_trans = result
            .dimension_translations
            .iter()
            .find(|d| d.dimension == KVectorDimension::Integrity)
            .unwrap();

        // Integrity is valued in both domains, should transfer reasonably
        assert!(integrity_trans.translation_factor > 0.5);

        // Find topology translation (should transfer poorly - high in social, low in financial)
        let topology_trans = result
            .dimension_translations
            .iter()
            .find(|d| d.dimension == KVectorDimension::Topology)
            .unwrap();

        // Topology much less relevant in financial
        assert!(topology_trans.target_relevance < topology_trans.source_relevance);
    }

    #[test]
    fn test_similarity_matrix() {
        let registry = DomainRegistry::with_defaults();
        let matrix = registry.similarity_matrix();

        // Should have entries for all domain pairs
        let domain_count = registry.list_domains().len();
        assert_eq!(matrix.len(), domain_count * domain_count);

        // Diagonal should be 1.0 (self-similarity)
        assert!(
            (matrix
                .get(&("financial".to_string(), "financial".to_string()))
                .unwrap()
                - 1.0)
                .abs()
                < 0.001
        );
    }

    #[test]
    fn test_compute_domain_trust() {
        // Use a K-Vector with varied values to highlight domain differences
        // High stake (financial) but low topology (social contrast)
        let kv = KVector::new(
            0.5, // k_r: reputation (moderate)
            0.3, // k_a: activity (low)
            0.9, // k_i: integrity (high)
            0.4, // k_p: performance (moderate)
            0.6, // k_m: membership (moderate)
            0.9, // k_s: stake (high - financial loves this)
            0.7, // k_h: historical (high)
            0.2, // k_topo: topology (low - social needs this)
            0.8, // k_v: verification (high)
            0.6, // k_coherence: coherence (moderate)
        );

        let financial = DomainTemplates::financial();
        let social = DomainTemplates::social();

        let fin_trust = compute_domain_trust(&kv, &financial);
        let soc_trust = compute_domain_trust(&kv, &social);

        // Both should be valid trust scores
        assert!(fin_trust >= 0.0 && fin_trust <= 1.0);
        assert!(soc_trust >= 0.0 && soc_trust <= 1.0);

        // Financial should be higher (high stake, high verification)
        // Social should be lower (low topology, low activity)
        assert!(
            fin_trust > soc_trust,
            "Expected financial ({:.3}) > social ({:.3}) for high-stake low-topology agent",
            fin_trust,
            soc_trust
        );
    }
}
