// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claim indexing for fast queries

use crate::claims::{DesciClaim, EpistemicTier};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// In-memory index for claims
#[derive(Debug, Clone)]
pub struct ClaimIndex {
    /// Index by category
    category_index: HashMap<String, HashSet<Uuid>>,

    /// Index by tier
    tier_index: HashMap<EpistemicTier, HashSet<Uuid>>,

    /// Keyword inverted index
    keyword_index: HashMap<String, HashSet<Uuid>>,

    /// Index by creator
    creator_index: HashMap<String, HashSet<Uuid>>,

    /// All claim IDs (for full scans)
    all_claims: HashSet<Uuid>,
}

impl ClaimIndex {
    /// Create a new empty index
    pub fn new() -> Self {
        Self {
            category_index: HashMap::new(),
            tier_index: HashMap::new(),
            keyword_index: HashMap::new(),
            creator_index: HashMap::new(),
            all_claims: HashSet::new(),
        }
    }

    /// Add a claim to the index
    pub fn add_claim(&mut self, claim: &DesciClaim) {
        let claim_id = claim.id;

        // Add to all claims
        self.all_claims.insert(claim_id);

        // Index by category (normalized to lowercase)
        let category = claim.content.category.to_lowercase();
        self.category_index
            .entry(category)
            .or_insert_with(HashSet::new)
            .insert(claim_id);

        // Index by tier
        self.tier_index
            .entry(claim.epistemic_tier)
            .or_insert_with(HashSet::new)
            .insert(claim_id);

        // Index by keywords (normalized to lowercase)
        for keyword in &claim.content.keywords {
            let normalized = keyword.to_lowercase();
            self.keyword_index
                .entry(normalized)
                .or_insert_with(HashSet::new)
                .insert(claim_id);
        }

        // Index by creator
        self.creator_index
            .entry(claim.creator.clone())
            .or_insert_with(HashSet::new)
            .insert(claim_id);
    }

    /// Remove a claim from the index
    pub fn remove_claim(&mut self, claim: &DesciClaim) {
        let claim_id = claim.id;

        self.all_claims.remove(&claim_id);

        // Remove from category index
        let category = claim.content.category.to_lowercase();
        if let Some(set) = self.category_index.get_mut(&category) {
            set.remove(&claim_id);
        }

        // Remove from tier index
        if let Some(set) = self.tier_index.get_mut(&claim.epistemic_tier) {
            set.remove(&claim_id);
        }

        // Remove from keyword index
        for keyword in &claim.content.keywords {
            let normalized = keyword.to_lowercase();
            if let Some(set) = self.keyword_index.get_mut(&normalized) {
                set.remove(&claim_id);
            }
        }

        // Remove from creator index
        if let Some(set) = self.creator_index.get_mut(&claim.creator) {
            set.remove(&claim_id);
        }
    }

    /// Get claim IDs by category (partial match)
    pub fn by_category(&self, category: &str) -> HashSet<Uuid> {
        let normalized = category.to_lowercase();
        let mut results = HashSet::new();

        for (cat, ids) in &self.category_index {
            if cat.contains(&normalized) {
                results.extend(ids);
            }
        }

        results
    }

    /// Get claim IDs by tier
    pub fn by_tier(&self, tier: EpistemicTier) -> HashSet<Uuid> {
        self.tier_index.get(&tier).cloned().unwrap_or_default()
    }

    /// Get claim IDs by minimum tier
    pub fn by_min_tier(&self, min_tier: EpistemicTier) -> HashSet<Uuid> {
        let mut results = HashSet::new();

        for (tier, ids) in &self.tier_index {
            if *tier >= min_tier {
                results.extend(ids);
            }
        }

        results
    }

    /// Get claim IDs by keyword (partial match)
    pub fn by_keyword(&self, keyword: &str) -> HashSet<Uuid> {
        let normalized = keyword.to_lowercase();
        let mut results = HashSet::new();

        for (kw, ids) in &self.keyword_index {
            if kw.contains(&normalized) {
                results.extend(ids);
            }
        }

        results
    }

    /// Get claim IDs by creator
    pub fn by_creator(&self, creator: &str) -> HashSet<Uuid> {
        self.creator_index.get(creator).cloned().unwrap_or_default()
    }

    /// Get all claim IDs
    pub fn all(&self) -> HashSet<Uuid> {
        self.all_claims.clone()
    }

    /// Get the number of indexed claims
    pub fn len(&self) -> usize {
        self.all_claims.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.all_claims.is_empty()
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.category_index.clear();
        self.tier_index.clear();
        self.keyword_index.clear();
        self.creator_index.clear();
        self.all_claims.clear();
    }

    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            total_claims: self.all_claims.len(),
            unique_categories: self.category_index.len(),
            unique_keywords: self.keyword_index.len(),
            unique_creators: self.creator_index.len(),
        }
    }
}

impl Default for ClaimIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub total_claims: usize,
    pub unique_categories: usize,
    pub unique_keywords: usize,
    pub unique_creators: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::ClaimContent;

    fn create_test_claim(category: &str, tier: EpistemicTier, keywords: Vec<&str>) -> DesciClaim {
        let content = ClaimContent {
            dataset_hash: "test".to_string(),
            description: "Test".to_string(),
            category: category.to_string(),
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
            storage_ref: None,
            reproducibility_score: None,
            license: None,
        };

        DesciClaim::new(tier, content, "creator".to_string())
    }

    #[test]
    fn test_add_and_retrieve() {
        let mut index = ClaimIndex::new();
        let claim = create_test_claim("genomics", EpistemicTier::E2, vec!["CRISPR"]);

        index.add_claim(&claim);

        assert_eq!(index.len(), 1);
        assert!(index.all().contains(&claim.id));
    }

    #[test]
    fn test_category_index() {
        let mut index = ClaimIndex::new();
        let claim1 = create_test_claim("genomics", EpistemicTier::E2, vec![]);
        let claim2 = create_test_claim("genomics", EpistemicTier::E3, vec![]);
        let claim3 = create_test_claim("climate", EpistemicTier::E1, vec![]);

        index.add_claim(&claim1);
        index.add_claim(&claim2);
        index.add_claim(&claim3);

        let genomics_claims = index.by_category("genomics");
        assert_eq!(genomics_claims.len(), 2);
        assert!(genomics_claims.contains(&claim1.id));
        assert!(genomics_claims.contains(&claim2.id));
    }

    #[test]
    fn test_tier_index() {
        let mut index = ClaimIndex::new();
        let claim1 = create_test_claim("test", EpistemicTier::E2, vec![]);
        let claim2 = create_test_claim("test", EpistemicTier::E2, vec![]);
        let claim3 = create_test_claim("test", EpistemicTier::E3, vec![]);

        index.add_claim(&claim1);
        index.add_claim(&claim2);
        index.add_claim(&claim3);

        let e2_claims = index.by_tier(EpistemicTier::E2);
        assert_eq!(e2_claims.len(), 2);
    }

    #[test]
    fn test_min_tier_index() {
        let mut index = ClaimIndex::new();
        let claim1 = create_test_claim("test", EpistemicTier::E1, vec![]);
        let claim2 = create_test_claim("test", EpistemicTier::E2, vec![]);
        let claim3 = create_test_claim("test", EpistemicTier::E3, vec![]);

        index.add_claim(&claim1);
        index.add_claim(&claim2);
        index.add_claim(&claim3);

        let min_e2 = index.by_min_tier(EpistemicTier::E2);
        assert_eq!(min_e2.len(), 2);  // E2 and E3
        assert!(!min_e2.contains(&claim1.id));
    }

    #[test]
    fn test_keyword_index() {
        let mut index = ClaimIndex::new();
        let claim1 = create_test_claim("test", EpistemicTier::E1, vec!["CRISPR", "gene-editing"]);
        let claim2 = create_test_claim("test", EpistemicTier::E2, vec!["CRISPR", "cancer"]);
        let claim3 = create_test_claim("test", EpistemicTier::E3, vec!["NAD+"]);

        index.add_claim(&claim1);
        index.add_claim(&claim2);
        index.add_claim(&claim3);

        let crispr_claims = index.by_keyword("CRISPR");
        assert_eq!(crispr_claims.len(), 2);

        let nad_claims = index.by_keyword("NAD+");
        assert_eq!(nad_claims.len(), 1);
    }

    #[test]
    fn test_remove_claim() {
        let mut index = ClaimIndex::new();
        let claim = create_test_claim("genomics", EpistemicTier::E2, vec!["CRISPR"]);

        index.add_claim(&claim);
        assert_eq!(index.len(), 1);

        index.remove_claim(&claim);
        assert_eq!(index.len(), 0);
        assert!(!index.all().contains(&claim.id));
    }

    #[test]
    fn test_case_insensitive() {
        let mut index = ClaimIndex::new();
        let claim = create_test_claim("Genomics", EpistemicTier::E2, vec!["CRISPR"]);

        index.add_claim(&claim);

        // Should match regardless of case
        assert!(!index.by_category("genomics").is_empty());
        assert!(!index.by_category("GENOMICS").is_empty());
        assert!(!index.by_keyword("crispr").is_empty());
    }
}
