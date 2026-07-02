// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Expertise Domains & Weighted Verification
//!
//! Tracks domain-specific expertise for verifiers and applies weighted
//! verification multipliers based on expertise alignment with claim domains.

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// A domain of expertise (hierarchical)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExpertiseDomain {
    /// Unique identifier
    pub id: Uuid,
    /// Human-readable name
    pub name: String,
    /// Parent domain (for hierarchy)
    pub parent: Option<Uuid>,
    /// Keywords associated with this domain
    pub keywords: Vec<String>,
    /// Depth in hierarchy (0 = root)
    pub depth: usize,
}

impl ExpertiseDomain {
    /// Create a new root domain
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            parent: None,
            keywords: Vec::new(),
            depth: 0,
        }
    }

    /// Create a child domain
    pub fn child_of(name: impl Into<String>, parent: &ExpertiseDomain) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            parent: Some(parent.id),
            keywords: Vec::new(),
            depth: parent.depth + 1,
        }
    }

    /// Add keywords to the domain
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = keywords;
        self
    }
}

/// Expertise level for a verifier in a domain
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertiseLevel {
    /// Core competency score (0.0-1.0)
    pub competency: f64,
    /// Experience metric (years or equivalent)
    pub experience: f64,
    /// Number of successful verifications in this domain
    pub verification_count: usize,
    /// Accuracy rate in this domain
    pub accuracy_rate: f64,
    /// Last activity timestamp
    pub last_active: i64,
    /// Whether credentials are verified
    pub credentials_verified: bool,
}

impl Default for ExpertiseLevel {
    fn default() -> Self {
        Self {
            competency: 0.0,
            experience: 0.0,
            verification_count: 0,
            accuracy_rate: 0.0,
            last_active: 0,
            credentials_verified: false,
        }
    }
}

impl ExpertiseLevel {
    /// Calculate overall expertise score
    pub fn score(&self) -> f64 {
        let base = (self.competency * 0.4)
            + ((self.experience / 10.0).min(1.0) * 0.2)
            + (self.accuracy_rate * 0.3)
            + ((self.verification_count as f64 / 100.0).min(1.0) * 0.1);

        // Credential bonus
        let credential_multiplier = if self.credentials_verified { 1.2 } else { 1.0 };

        (base * credential_multiplier).min(1.0)
    }
}

/// A verifier's expertise profile
#[derive(Debug, Clone)]
pub struct ExpertiseProfile {
    /// Verifier ID
    pub verifier_id: Uuid,
    /// Expertise levels by domain
    pub domains: HashMap<Uuid, ExpertiseLevel>,
    /// Overall reputation score
    pub reputation: f64,
    /// Total verifications across all domains
    pub total_verifications: usize,
    /// Cross-domain versatility score
    pub versatility: f64,
}

impl ExpertiseProfile {
    /// Create a new empty profile
    pub fn new(verifier_id: Uuid) -> Self {
        Self {
            verifier_id,
            domains: HashMap::new(),
            reputation: 0.5, // Start neutral
            total_verifications: 0,
            versatility: 0.0,
        }
    }

    /// Get expertise level for a domain
    pub fn get_expertise(&self, domain_id: Uuid) -> Option<&ExpertiseLevel> {
        self.domains.get(&domain_id)
    }

    /// Set expertise level for a domain
    pub fn set_expertise(&mut self, domain_id: Uuid, level: ExpertiseLevel) {
        self.domains.insert(domain_id, level);
        self.recalculate_versatility();
    }

    /// Update expertise after a verification
    pub fn record_verification(
        &mut self,
        domain_id: Uuid,
        was_accurate: bool,
        timestamp: i64,
    ) {
        let level = self.domains.entry(domain_id).or_default();
        level.verification_count += 1;
        level.last_active = timestamp;

        // Update accuracy rate with exponential moving average
        let alpha = 0.1;
        let new_accuracy = if was_accurate { 1.0 } else { 0.0 };
        level.accuracy_rate = (alpha * new_accuracy) + ((1.0 - alpha) * level.accuracy_rate);

        // Slowly increase competency with experience
        level.competency = (level.competency + 0.01).min(1.0);

        self.total_verifications += 1;
        self.recalculate_versatility();
    }

    /// Recalculate versatility score
    fn recalculate_versatility(&mut self) {
        let domain_count = self.domains.len() as f64;
        if domain_count == 0.0 {
            self.versatility = 0.0;
            return;
        }

        // Versatility = normalized count of high-competency domains
        let high_competency_count = self
            .domains
            .values()
            .filter(|l| l.competency > 0.5)
            .count() as f64;

        self.versatility = (high_competency_count / domain_count.max(1.0)).min(1.0);
    }
}

/// Domain taxonomy for expertise classification
#[derive(Debug, Clone)]
pub struct DomainTaxonomy {
    /// All domains
    domains: HashMap<Uuid, ExpertiseDomain>,
    /// Root domains
    roots: Vec<Uuid>,
    /// Children by parent
    children: HashMap<Uuid, Vec<Uuid>>,
}

impl DomainTaxonomy {
    /// Create a new empty taxonomy
    pub fn new() -> Self {
        Self {
            domains: HashMap::new(),
            roots: Vec::new(),
            children: HashMap::new(),
        }
    }

    /// Add a domain to the taxonomy
    pub fn add_domain(&mut self, domain: ExpertiseDomain) {
        let id = domain.id;
        let parent = domain.parent;

        self.domains.insert(id, domain);

        if let Some(parent_id) = parent {
            self.children.entry(parent_id).or_default().push(id);
        } else {
            self.roots.push(id);
        }
    }

    /// Get a domain by ID
    pub fn get_domain(&self, id: Uuid) -> Option<&ExpertiseDomain> {
        self.domains.get(&id)
    }

    /// Get all ancestors of a domain
    pub fn get_ancestors(&self, id: Uuid) -> Vec<Uuid> {
        let mut ancestors = Vec::new();
        let mut current = id;

        while let Some(domain) = self.domains.get(&current) {
            if let Some(parent_id) = domain.parent {
                ancestors.push(parent_id);
                current = parent_id;
            } else {
                break;
            }
        }

        ancestors
    }

    /// Get all descendants of a domain
    pub fn get_descendants(&self, id: Uuid) -> Vec<Uuid> {
        let mut descendants = Vec::new();
        let mut queue = vec![id];

        while let Some(current) = queue.pop() {
            if let Some(children) = self.children.get(&current) {
                for &child in children {
                    descendants.push(child);
                    queue.push(child);
                }
            }
        }

        descendants
    }

    /// Find domains matching keywords
    pub fn find_by_keywords(&self, keywords: &[String]) -> Vec<Uuid> {
        let keyword_set: HashSet<_> = keywords.iter().map(|k| k.to_lowercase()).collect();

        self.domains
            .iter()
            .filter(|(_, domain)| {
                domain
                    .keywords
                    .iter()
                    .any(|k| keyword_set.contains(&k.to_lowercase()))
            })
            .map(|(&id, _)| id)
            .collect()
    }

    /// Calculate similarity between two domains
    pub fn domain_similarity(&self, a: Uuid, b: Uuid) -> f64 {
        if a == b {
            return 1.0;
        }

        let ancestors_a: HashSet<_> = self.get_ancestors(a).into_iter().collect();
        let ancestors_b: HashSet<_> = self.get_ancestors(b).into_iter().collect();

        // Find lowest common ancestor
        let common: HashSet<_> = ancestors_a.intersection(&ancestors_b).cloned().collect();

        if common.is_empty() {
            // Check if one is ancestor of other
            if ancestors_a.contains(&b) || ancestors_b.contains(&a) {
                return 0.5;
            }
            return 0.0;
        }

        // Similarity based on distance to common ancestor
        let domain_a = self.domains.get(&a);
        let domain_b = self.domains.get(&b);

        match (domain_a, domain_b) {
            (Some(da), Some(db)) => {
                let depth_diff = (da.depth as i32 - db.depth as i32).unsigned_abs() as f64;
                (1.0 / (1.0 + depth_diff * 0.3)).min(0.8)
            }
            _ => 0.0,
        }
    }
}

impl Default for DomainTaxonomy {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for expertise-weighted verification
#[derive(Debug, Clone)]
pub struct ExpertiseWeightConfig {
    /// Minimum expertise score to verify
    pub min_expertise_threshold: f64,
    /// Maximum weight multiplier for experts
    pub max_expert_multiplier: f64,
    /// Weight for non-experts (below threshold)
    pub novice_weight: f64,
    /// Whether to consider related domains
    pub consider_related_domains: bool,
    /// Decay factor for related domain expertise
    pub related_domain_decay: f64,
    /// Minimum verifiers required
    pub min_verifiers: usize,
}

impl Default for ExpertiseWeightConfig {
    fn default() -> Self {
        Self {
            min_expertise_threshold: 0.3,
            max_expert_multiplier: 3.0,
            novice_weight: 0.5,
            consider_related_domains: true,
            related_domain_decay: 0.5,
            min_verifiers: 3,
        }
    }
}

/// Expertise-weighted verification calculator
pub struct ExpertiseVerifier {
    config: ExpertiseWeightConfig,
}

impl ExpertiseVerifier {
    /// Create with default config
    pub fn new() -> Self {
        Self {
            config: ExpertiseWeightConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: ExpertiseWeightConfig) -> Self {
        Self { config }
    }

    /// Calculate verification weight for a verifier in a domain
    pub fn calculate_weight(
        &self,
        profile: &ExpertiseProfile,
        claim_domain: Uuid,
        taxonomy: &DomainTaxonomy,
    ) -> f64 {
        // Direct domain expertise
        let direct_score = profile
            .get_expertise(claim_domain)
            .map(|l| l.score())
            .unwrap_or(0.0);

        if direct_score >= self.config.min_expertise_threshold {
            // Expert weight
            let normalized = (direct_score - self.config.min_expertise_threshold)
                / (1.0 - self.config.min_expertise_threshold);
            return 1.0 + (normalized * (self.config.max_expert_multiplier - 1.0));
        }

        if !self.config.consider_related_domains {
            return self.config.novice_weight;
        }

        // Check related domains
        let ancestors = taxonomy.get_ancestors(claim_domain);
        let descendants = taxonomy.get_descendants(claim_domain);

        let mut best_related_score: f64 = 0.0;
        let mut decay_factor = 1.0;

        // Check ancestors (parent domains)
        for ancestor in &ancestors {
            decay_factor *= self.config.related_domain_decay;
            if let Some(level) = profile.get_expertise(*ancestor) {
                let adjusted = level.score() * decay_factor;
                best_related_score = best_related_score.max(adjusted);
            }
        }

        // Check descendants (sub-domains)
        decay_factor = 1.0;
        for descendant in &descendants {
            decay_factor *= self.config.related_domain_decay;
            if let Some(level) = profile.get_expertise(*descendant) {
                let adjusted = level.score() * decay_factor;
                best_related_score = best_related_score.max(adjusted);
            }
        }

        if best_related_score >= self.config.min_expertise_threshold {
            // Partial expert weight from related domain
            let normalized = (best_related_score - self.config.min_expertise_threshold)
                / (1.0 - self.config.min_expertise_threshold);
            return 1.0 + (normalized * (self.config.max_expert_multiplier - 1.0) * 0.5);
        }

        self.config.novice_weight
    }

    /// Calculate weighted verification score for a claim
    pub fn calculate_weighted_score(
        &self,
        verifications: &[(Uuid, bool)], // (verifier_id, supports_claim)
        profiles: &HashMap<Uuid, ExpertiseProfile>,
        claim_domain: Uuid,
        taxonomy: &DomainTaxonomy,
    ) -> WeightedVerificationResult {
        let mut total_weight = 0.0;
        let mut support_weight = 0.0;
        let mut expert_count = 0;
        let mut novice_count = 0;

        for (verifier_id, supports) in verifications {
            let weight = match profiles.get(verifier_id) {
                Some(profile) => {
                    let w = self.calculate_weight(profile, claim_domain, taxonomy);
                    if w > 1.0 {
                        expert_count += 1;
                    } else {
                        novice_count += 1;
                    }
                    w
                }
                None => {
                    novice_count += 1;
                    self.config.novice_weight
                }
            };

            total_weight += weight;
            if *supports {
                support_weight += weight;
            }
        }

        let consensus_score = if total_weight > 0.0 {
            support_weight / total_weight
        } else {
            0.5
        };

        let confidence = if verifications.len() >= self.config.min_verifiers {
            let expert_ratio = expert_count as f64 / verifications.len() as f64;
            (0.5 + expert_ratio * 0.5).min(1.0)
        } else {
            0.3
        };

        WeightedVerificationResult {
            consensus_score,
            total_weight,
            expert_count,
            novice_count,
            confidence,
            meets_threshold: verifications.len() >= self.config.min_verifiers,
        }
    }
}

impl Default for ExpertiseVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of weighted verification calculation
#[derive(Debug, Clone)]
pub struct WeightedVerificationResult {
    /// Weighted consensus score (0.0-1.0, >0.5 = supports)
    pub consensus_score: f64,
    /// Total weight of all verifications
    pub total_weight: f64,
    /// Number of expert verifiers
    pub expert_count: usize,
    /// Number of non-expert verifiers
    pub novice_count: usize,
    /// Confidence in the result
    pub confidence: f64,
    /// Whether minimum verifier threshold is met
    pub meets_threshold: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_taxonomy() -> DomainTaxonomy {
        let mut taxonomy = DomainTaxonomy::new();

        // Create hierarchy: Science -> Biology -> Genetics
        let science = ExpertiseDomain::new("Science")
            .with_keywords(vec!["research".into(), "scientific".into()]);
        let biology = ExpertiseDomain::child_of("Biology", &science)
            .with_keywords(vec!["life".into(), "organism".into()]);
        let genetics = ExpertiseDomain::child_of("Genetics", &biology)
            .with_keywords(vec!["dna".into(), "gene".into()]);

        taxonomy.add_domain(science);
        taxonomy.add_domain(biology);
        taxonomy.add_domain(genetics);

        taxonomy
    }

    #[test]
    fn test_expertise_level_scoring() {
        let level = ExpertiseLevel {
            competency: 0.8,
            experience: 5.0,
            verification_count: 50,
            accuracy_rate: 0.9,
            last_active: 1000,
            credentials_verified: true,
        };

        let score = level.score();
        assert!(score > 0.5);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_expertise_profile() {
        let mut profile = ExpertiseProfile::new(Uuid::new_v4());
        let domain_id = Uuid::new_v4();

        // Record some verifications
        profile.record_verification(domain_id, true, 1000);
        profile.record_verification(domain_id, true, 2000);
        profile.record_verification(domain_id, false, 3000);

        let level = profile.get_expertise(domain_id).unwrap();
        assert_eq!(level.verification_count, 3);
        assert!(level.accuracy_rate > 0.0);
    }

    #[test]
    fn test_domain_taxonomy() {
        let taxonomy = create_test_taxonomy();

        assert_eq!(taxonomy.roots.len(), 1);
        assert_eq!(taxonomy.domains.len(), 3);

        // Find genetics domain
        let genetics_results = taxonomy.find_by_keywords(&["gene".to_string()]);
        assert!(!genetics_results.is_empty());
    }

    #[test]
    fn test_expertise_weight_calculation() {
        let taxonomy = create_test_taxonomy();
        let verifier = ExpertiseVerifier::new();

        // Create an expert profile
        let mut profile = ExpertiseProfile::new(Uuid::new_v4());
        let domain_id = taxonomy.roots[0]; // Science

        profile.set_expertise(
            domain_id,
            ExpertiseLevel {
                competency: 0.9,
                experience: 10.0,
                verification_count: 100,
                accuracy_rate: 0.95,
                last_active: 1000,
                credentials_verified: true,
            },
        );

        let weight = verifier.calculate_weight(&profile, domain_id, &taxonomy);
        assert!(weight > 1.0); // Should be expert weight
    }

    #[test]
    fn test_weighted_verification() {
        let taxonomy = create_test_taxonomy();
        let verifier = ExpertiseVerifier::new();
        let domain_id = taxonomy.roots[0];

        // Create profiles
        let mut profiles = HashMap::new();

        let expert_id = Uuid::new_v4();
        let mut expert_profile = ExpertiseProfile::new(expert_id);
        expert_profile.set_expertise(
            domain_id,
            ExpertiseLevel {
                competency: 0.9,
                accuracy_rate: 0.95,
                credentials_verified: true,
                ..Default::default()
            },
        );
        profiles.insert(expert_id, expert_profile);

        let novice_id = Uuid::new_v4();
        let novice_profile = ExpertiseProfile::new(novice_id);
        profiles.insert(novice_id, novice_profile);

        // Verifications: expert supports, novice opposes
        let verifications = vec![
            (expert_id, true),
            (novice_id, false),
            (Uuid::new_v4(), true), // Unknown verifier
        ];

        let result =
            verifier.calculate_weighted_score(&verifications, &profiles, domain_id, &taxonomy);

        // Expert support should outweigh novice opposition
        assert!(result.consensus_score > 0.5);
        assert_eq!(result.expert_count, 1);
    }
}
