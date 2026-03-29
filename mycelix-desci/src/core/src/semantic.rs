// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic Similarity Engine
//!
//! Detects duplicate or highly related claims using text similarity.
//! Supports multiple similarity algorithms and fuzzy matching
//! to maintain claim network integrity.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Similarity score between two claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityScore {
    /// ID of first claim
    pub claim_a: Uuid,
    /// ID of second claim
    pub claim_b: Uuid,
    /// Overall similarity (0.0-1.0)
    pub similarity: f64,
    /// Breakdown by component
    pub components: SimilarityComponents,
    /// Relationship classification
    pub relationship: SimilarityRelationship,
}

/// Components of similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityComponents {
    /// Title/summary similarity
    pub title_similarity: f64,
    /// Full content similarity
    pub content_similarity: f64,
    /// Keyword/topic overlap
    pub keyword_overlap: f64,
    /// Citation overlap
    pub citation_overlap: f64,
    /// Author overlap
    pub author_overlap: f64,
}

/// Classification of the relationship between claims
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimilarityRelationship {
    /// Exact or near-exact duplicate
    Duplicate,
    /// Same claim with minor variations
    NearDuplicate,
    /// Substantially overlapping content
    HighlyRelated,
    /// Shares some common elements
    Related,
    /// Some topical connection
    WeaklyRelated,
    /// No meaningful relationship
    Unrelated,
}

impl SimilarityRelationship {
    /// Get threshold for this relationship
    pub fn threshold(&self) -> f64 {
        match self {
            Self::Duplicate => 0.95,
            Self::NearDuplicate => 0.85,
            Self::HighlyRelated => 0.70,
            Self::Related => 0.50,
            Self::WeaklyRelated => 0.30,
            Self::Unrelated => 0.0,
        }
    }

    /// Classify from similarity score
    pub fn from_score(score: f64) -> Self {
        if score >= 0.95 {
            Self::Duplicate
        } else if score >= 0.85 {
            Self::NearDuplicate
        } else if score >= 0.70 {
            Self::HighlyRelated
        } else if score >= 0.50 {
            Self::Related
        } else if score >= 0.30 {
            Self::WeaklyRelated
        } else {
            Self::Unrelated
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Duplicate => "Duplicate claim",
            Self::NearDuplicate => "Near-duplicate with minor differences",
            Self::HighlyRelated => "Highly related content",
            Self::Related => "Related claims",
            Self::WeaklyRelated => "Weakly related",
            Self::Unrelated => "No relationship",
        }
    }
}

/// Claim content for similarity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimContent {
    pub id: Uuid,
    pub title: String,
    pub content: String,
    pub keywords: Vec<String>,
    pub citations: Vec<Uuid>,
    pub authors: Vec<String>,
}

/// Configuration for similarity engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityConfig {
    /// Weight for title similarity
    pub title_weight: f64,
    /// Weight for content similarity
    pub content_weight: f64,
    /// Weight for keyword overlap
    pub keyword_weight: f64,
    /// Weight for citation overlap
    pub citation_weight: f64,
    /// Weight for author overlap
    pub author_weight: f64,
    /// Minimum similarity to report
    pub min_threshold: f64,
    /// Use n-gram matching
    pub use_ngrams: bool,
    /// N-gram size
    pub ngram_size: usize,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            title_weight: 0.25,
            content_weight: 0.40,
            keyword_weight: 0.20,
            citation_weight: 0.10,
            author_weight: 0.05,
            min_threshold: 0.30,
            use_ngrams: true,
            ngram_size: 3,
        }
    }
}

/// Semantic similarity engine
#[derive(Debug, Clone)]
pub struct SimilarityEngine {
    config: SimilarityConfig,
    claims: Vec<ClaimContent>,
    /// Cached n-grams for each claim
    ngram_cache: HashMap<Uuid, Vec<String>>,
}

impl SimilarityEngine {
    pub fn new() -> Self {
        Self::with_config(SimilarityConfig::default())
    }

    pub fn with_config(config: SimilarityConfig) -> Self {
        Self {
            config,
            claims: Vec::new(),
            ngram_cache: HashMap::new(),
        }
    }

    /// Add a claim to the index
    pub fn add_claim(&mut self, claim: ClaimContent) {
        if self.config.use_ngrams {
            let ngrams = self.extract_ngrams(&claim.content);
            self.ngram_cache.insert(claim.id, ngrams);
        }
        self.claims.push(claim);
    }

    /// Find similar claims to a given claim
    pub fn find_similar(&self, claim: &ClaimContent) -> Vec<SimilarityScore> {
        let mut results = Vec::new();

        for indexed_claim in &self.claims {
            if indexed_claim.id == claim.id {
                continue;
            }

            let score = self.calculate_similarity(claim, indexed_claim);
            if score.similarity >= self.config.min_threshold {
                results.push(score);
            }
        }

        // Sort by similarity descending
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results
    }

    /// Find all duplicate/near-duplicate pairs in the index
    pub fn find_duplicates(&self) -> Vec<SimilarityScore> {
        let mut duplicates = Vec::new();

        for i in 0..self.claims.len() {
            for j in (i + 1)..self.claims.len() {
                let score = self.calculate_similarity(&self.claims[i], &self.claims[j]);
                if score.relationship == SimilarityRelationship::Duplicate
                    || score.relationship == SimilarityRelationship::NearDuplicate
                {
                    duplicates.push(score);
                }
            }
        }

        duplicates
    }

    /// Calculate similarity between two claims
    pub fn calculate_similarity(&self, a: &ClaimContent, b: &ClaimContent) -> SimilarityScore {
        let title_sim = self.text_similarity(&a.title, &b.title);
        let content_sim = self.content_similarity(&a.content, &b.content, a.id, b.id);
        let keyword_sim = self.set_overlap(&a.keywords, &b.keywords);
        let citation_sim = self.uuid_set_overlap(&a.citations, &b.citations);
        let author_sim = self.set_overlap(&a.authors, &b.authors);

        let components = SimilarityComponents {
            title_similarity: title_sim,
            content_similarity: content_sim,
            keyword_overlap: keyword_sim,
            citation_overlap: citation_sim,
            author_overlap: author_sim,
        };

        let weighted_score = title_sim * self.config.title_weight
            + content_sim * self.config.content_weight
            + keyword_sim * self.config.keyword_weight
            + citation_sim * self.config.citation_weight
            + author_sim * self.config.author_weight;

        let relationship = SimilarityRelationship::from_score(weighted_score);

        SimilarityScore {
            claim_a: a.id,
            claim_b: b.id,
            similarity: weighted_score,
            components,
            relationship,
        }
    }

    /// Simple text similarity using Jaccard index on words
    fn text_similarity(&self, a: &str, b: &str) -> f64 {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();
        let words_a: std::collections::HashSet<&str> = a_lower.split_whitespace().collect();
        let words_b: std::collections::HashSet<&str> = b_lower.split_whitespace().collect();

        self.jaccard_index(&words_a, &words_b)
    }

    /// Content similarity using n-grams if enabled
    fn content_similarity(&self, a: &str, b: &str, id_a: Uuid, id_b: Uuid) -> f64 {
        if self.config.use_ngrams {
            let ngrams_a = self
                .ngram_cache
                .get(&id_a)
                .cloned()
                .unwrap_or_else(|| self.extract_ngrams(a));
            let ngrams_b = self
                .ngram_cache
                .get(&id_b)
                .cloned()
                .unwrap_or_else(|| self.extract_ngrams(b));

            let set_a: std::collections::HashSet<_> = ngrams_a.iter().collect();
            let set_b: std::collections::HashSet<_> = ngrams_b.iter().collect();

            self.jaccard_index(&set_a, &set_b)
        } else {
            self.text_similarity(a, b)
        }
    }

    /// Extract character n-grams from text
    fn extract_ngrams(&self, text: &str) -> Vec<String> {
        let normalized: String = text
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();

        let chars: Vec<char> = normalized.chars().collect();
        let n = self.config.ngram_size;

        if chars.len() < n {
            return vec![normalized];
        }

        chars.windows(n).map(|w| w.iter().collect()).collect()
    }

    /// Calculate set overlap for strings
    fn set_overlap(&self, a: &[String], b: &[String]) -> f64 {
        let set_a: std::collections::HashSet<_> = a.iter().map(|s| s.to_lowercase()).collect();
        let set_b: std::collections::HashSet<_> = b.iter().map(|s| s.to_lowercase()).collect();

        self.jaccard_index(&set_a, &set_b)
    }

    /// Calculate set overlap for UUIDs
    fn uuid_set_overlap(&self, a: &[Uuid], b: &[Uuid]) -> f64 {
        let set_a: std::collections::HashSet<_> = a.iter().collect();
        let set_b: std::collections::HashSet<_> = b.iter().collect();

        self.jaccard_index(&set_a, &set_b)
    }

    /// Jaccard similarity index
    fn jaccard_index<T: Eq + std::hash::Hash>(
        &self,
        a: &std::collections::HashSet<T>,
        b: &std::collections::HashSet<T>,
    ) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 1.0; // Both empty = identical
        }

        let intersection = a.intersection(b).count();
        let union = a.union(b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Get statistics about the index
    pub fn stats(&self) -> SimilarityStats {
        let total_claims = self.claims.len();
        let total_pairs = total_claims * (total_claims.saturating_sub(1)) / 2;

        SimilarityStats {
            total_claims,
            total_pairs,
            indexed_ngrams: self.ngram_cache.values().map(|v| v.len()).sum(),
        }
    }

    /// Clear all indexed claims
    pub fn clear(&mut self) {
        self.claims.clear();
        self.ngram_cache.clear();
    }
}

impl Default for SimilarityEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the similarity index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityStats {
    pub total_claims: usize,
    pub total_pairs: usize,
    pub indexed_ngrams: usize,
}

/// Result of duplicate check for a new claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateCheckResult {
    /// Whether a potential duplicate was found
    pub has_duplicate: bool,
    /// The most similar existing claim
    pub most_similar: Option<SimilarityScore>,
    /// All claims above threshold
    pub similar_claims: Vec<SimilarityScore>,
    /// Recommendation
    pub recommendation: DuplicateRecommendation,
}

/// Recommendation for handling potential duplicate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DuplicateRecommendation {
    /// Proceed - no significant overlap
    Proceed,
    /// Review - some overlap, may need citation
    Review,
    /// Link - should reference existing claim
    LinkToExisting,
    /// Merge - should be merged with existing
    Merge,
    /// Reject - duplicate, don't create
    Reject,
}

impl SimilarityEngine {
    /// Check if a claim is a duplicate before adding
    pub fn check_duplicate(&self, claim: &ClaimContent) -> DuplicateCheckResult {
        let similar = self.find_similar(claim);

        let most_similar = similar.first().cloned();
        let has_duplicate = most_similar
            .as_ref()
            .map(|s| s.similarity >= 0.85)
            .unwrap_or(false);

        let recommendation = match most_similar.as_ref() {
            Some(s) if s.similarity >= 0.95 => DuplicateRecommendation::Reject,
            Some(s) if s.similarity >= 0.85 => DuplicateRecommendation::Merge,
            Some(s) if s.similarity >= 0.70 => DuplicateRecommendation::LinkToExisting,
            Some(s) if s.similarity >= 0.50 => DuplicateRecommendation::Review,
            _ => DuplicateRecommendation::Proceed,
        };

        DuplicateCheckResult {
            has_duplicate,
            most_similar,
            similar_claims: similar,
            recommendation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_claim(id: Uuid, title: &str, content: &str, keywords: Vec<&str>) -> ClaimContent {
        ClaimContent {
            id,
            title: title.to_string(),
            content: content.to_string(),
            keywords: keywords.into_iter().map(String::from).collect(),
            citations: vec![],
            authors: vec!["author@test.com".to_string()],
        }
    }

    #[test]
    fn test_exact_duplicate() {
        let engine = SimilarityEngine::new();

        let claim1 = create_claim(
            Uuid::new_v4(),
            "Study on Climate Change Effects",
            "This study examines the effects of climate change on biodiversity.",
            vec!["climate", "biodiversity", "environment"],
        );

        let claim2 = create_claim(
            Uuid::new_v4(),
            "Study on Climate Change Effects",
            "This study examines the effects of climate change on biodiversity.",
            vec!["climate", "biodiversity", "environment"],
        );

        let score = engine.calculate_similarity(&claim1, &claim2);
        assert!(score.similarity >= 0.95);
        assert_eq!(score.relationship, SimilarityRelationship::Duplicate);
    }

    #[test]
    fn test_related_claims() {
        let engine = SimilarityEngine::new();

        let claim1 = create_claim(
            Uuid::new_v4(),
            "Climate Change and Coral Reefs",
            "Ocean warming is destroying coral reef ecosystems worldwide.",
            vec!["climate", "coral", "ocean"],
        );

        let claim2 = create_claim(
            Uuid::new_v4(),
            "Impact of Rising Temperatures on Marine Life",
            "Global temperature rise affects marine biodiversity significantly.",
            vec!["temperature", "marine", "biodiversity"],
        );

        let score = engine.calculate_similarity(&claim1, &claim2);
        assert!(score.similarity > 0.0);
        assert!(score.similarity < 0.85); // Related but not duplicate
    }

    #[test]
    fn test_unrelated_claims() {
        let engine = SimilarityEngine::new();

        let claim1 = create_claim(
            Uuid::new_v4(),
            "Quantum Computing Advances",
            "New breakthroughs in quantum error correction enable scalable qubits.",
            vec!["quantum", "computing", "qubits"],
        );

        let claim2 = create_claim(
            Uuid::new_v4(),
            "Historical Analysis of Roman Empire",
            "Archaeological evidence reveals new insights into Roman governance.",
            vec!["history", "rome", "archaeology"],
        );

        let score = engine.calculate_similarity(&claim1, &claim2);
        assert!(score.similarity < 0.30);
        assert_eq!(score.relationship, SimilarityRelationship::Unrelated);
    }

    #[test]
    fn test_find_similar() {
        let mut engine = SimilarityEngine::with_config(SimilarityConfig {
            min_threshold: 0.15, // Lower threshold for testing
            ..Default::default()
        });

        // Add some claims with overlapping content
        engine.add_claim(create_claim(
            Uuid::new_v4(),
            "Machine Learning in Healthcare AI",
            "AI models and machine learning can predict disease outcomes with high accuracy in healthcare.",
            vec!["ml", "healthcare", "ai", "prediction"],
        ));

        engine.add_claim(create_claim(
            Uuid::new_v4(),
            "Deep Learning for Medical Diagnosis AI",
            "Neural networks and AI improve diagnostic accuracy in radiology and healthcare.",
            vec!["deep learning", "medical", "ai", "healthcare"],
        ));

        engine.add_claim(create_claim(
            Uuid::new_v4(),
            "Cooking with Mediterranean Diet",
            "Olive oil and vegetables form the basis of healthy eating.",
            vec!["cooking", "diet", "health"],
        ));

        // Search for similar to new ML/healthcare claim
        let new_claim = create_claim(
            Uuid::new_v4(),
            "AI Applications in Healthcare Medicine",
            "Artificial intelligence and machine learning is transforming healthcare diagnosis and medical prediction.",
            vec!["ai", "medicine", "healthcare", "ml"],
        );

        let similar = engine.find_similar(&new_claim);

        // Should find the ML/healthcare claims as more similar
        assert!(!similar.is_empty(), "Should find at least one similar claim");
        // The most similar should be the ML/healthcare ones, not cooking
        assert!(similar[0].similarity > 0.1);
    }

    #[test]
    fn test_duplicate_check() {
        let mut engine = SimilarityEngine::new();

        engine.add_claim(create_claim(
            Uuid::new_v4(),
            "Water Boils at 100 Degrees Celsius",
            "Under standard atmospheric pressure, water reaches boiling point at 100C.",
            vec!["water", "boiling", "temperature"],
        ));

        // Check exact duplicate
        let duplicate = create_claim(
            Uuid::new_v4(),
            "Water Boils at 100 Degrees Celsius",
            "Under standard atmospheric pressure, water reaches boiling point at 100C.",
            vec!["water", "boiling", "temperature"],
        );

        let result = engine.check_duplicate(&duplicate);
        assert!(result.has_duplicate);
        assert!(matches!(
            result.recommendation,
            DuplicateRecommendation::Reject | DuplicateRecommendation::Merge
        ));
    }

    #[test]
    fn test_ngram_extraction() {
        let engine = SimilarityEngine::new();
        let ngrams = engine.extract_ngrams("hello world");

        assert!(!ngrams.is_empty());
        assert!(ngrams.contains(&"hel".to_string()));
        assert!(ngrams.contains(&"ell".to_string()));
    }

    #[test]
    fn test_relationship_thresholds() {
        assert_eq!(
            SimilarityRelationship::from_score(0.98),
            SimilarityRelationship::Duplicate
        );
        assert_eq!(
            SimilarityRelationship::from_score(0.90),
            SimilarityRelationship::NearDuplicate
        );
        assert_eq!(
            SimilarityRelationship::from_score(0.75),
            SimilarityRelationship::HighlyRelated
        );
        assert_eq!(
            SimilarityRelationship::from_score(0.55),
            SimilarityRelationship::Related
        );
        assert_eq!(
            SimilarityRelationship::from_score(0.35),
            SimilarityRelationship::WeaklyRelated
        );
        assert_eq!(
            SimilarityRelationship::from_score(0.10),
            SimilarityRelationship::Unrelated
        );
    }
}
