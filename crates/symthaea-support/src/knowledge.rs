// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Knowledge Manager — Article management, graduation, and cognitive update absorption

use crate::types::*;

#[derive(Debug)]
pub struct KnowledgeManager {
    articles: Vec<StoredArticle>,
}

#[derive(Debug, Clone)]
struct StoredArticle {
    id: String,
    title: String,
    category: SupportCategory,
    phi: f64,
    consolidation_strength: f64,
    #[allow(dead_code)]
    retrieval_count: u32,
    deprecated: bool,
    local_only: bool,
}

impl KnowledgeManager {
    pub fn new() -> Self {
        Self {
            articles: Vec::new(),
        }
    }

    /// Evaluate whether a resolution should be graduated to shared knowledge.
    pub fn evaluate_for_graduation(
        &self,
        phi: f64,
        effectiveness_rating: u8,
        retrieval_count: u32,
    ) -> GraduationDecision {
        if phi > 0.5 && effectiveness_rating >= 4 && retrieval_count >= 3 {
            GraduationDecision::Graduate
        } else if phi > 0.3 || retrieval_count >= 2 {
            GraduationDecision::Defer("Needs more evidence before graduation".to_string())
        } else {
            GraduationDecision::Reject("Insufficient quality or usage for graduation".to_string())
        }
    }

    /// Search knowledge base by keyword matching.
    pub fn search(&self, query: &str, k: usize) -> Vec<KnowledgeMatch> {
        let query_lower = query.to_lowercase();
        let mut matches: Vec<KnowledgeMatch> = self
            .articles
            .iter()
            .filter(|a| !a.deprecated)
            .filter(|a| a.title.to_lowercase().contains(&query_lower))
            .map(|a| KnowledgeMatch {
                article_id: a.id.clone(),
                title: a.title.clone(),
                similarity: a.consolidation_strength as f32,
                phi: a.phi,
            })
            .collect();
        matches.sort_by(|a, b| {
            b.phi
                .partial_cmp(&a.phi)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(k);
        matches
    }

    /// Search knowledge base, optionally filtering out local-only articles.
    ///
    /// When `can_share` is false, articles marked `local_only` are included.
    /// When `can_share` is true, articles marked `local_only` are excluded
    /// (they should not appear in federated results).
    pub fn search_filtered(&self, query: &str, k: usize, can_share: bool) -> Vec<KnowledgeMatch> {
        let query_lower = query.to_lowercase();
        let mut matches: Vec<KnowledgeMatch> = self
            .articles
            .iter()
            .filter(|a| !a.deprecated)
            .filter(|a| !can_share || !a.local_only)
            .filter(|a| a.title.to_lowercase().contains(&query_lower))
            .map(|a| KnowledgeMatch {
                article_id: a.id.clone(),
                title: a.title.clone(),
                similarity: a.consolidation_strength as f32,
                phi: a.phi,
            })
            .collect();
        matches.sort_by(|a, b| {
            b.phi
                .partial_cmp(&a.phi)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(k);
        matches
    }

    /// Add an article to the knowledge store.
    pub fn add_article(&mut self, id: String, title: String, category: SupportCategory, phi: f64) {
        self.articles.push(StoredArticle {
            id,
            title,
            category,
            phi,
            consolidation_strength: 1.0,
            retrieval_count: 0,
            deprecated: false,
            local_only: false,
        });
    }

    /// Add a local-only article that will not appear in federated search results.
    pub fn add_article_local_only(
        &mut self,
        id: String,
        title: String,
        category: SupportCategory,
        phi: f64,
    ) {
        self.articles.push(StoredArticle {
            id,
            title,
            category,
            phi,
            consolidation_strength: 1.0,
            retrieval_count: 0,
            deprecated: false,
            local_only: true,
        });
    }

    /// Deprecate an article, reducing its consolidation_strength.
    pub fn deprecate(&mut self, article_id: &str) -> bool {
        if let Some(article) = self.articles.iter_mut().find(|a| a.id == article_id) {
            article.deprecated = true;
            article.consolidation_strength *= 0.1;
            true
        } else {
            false
        }
    }

    /// Absorb a cognitive update from the DHT.
    pub fn absorb_cognitive_update(&mut self, update: &CognitiveUpdateData) -> AbsorptionResult {
        // Check for duplicate patterns
        let existing = self
            .articles
            .iter()
            .any(|a| a.category == update.category && !a.deprecated);
        let similarity = if existing { 0.5 } else { 0.0 };

        AbsorptionResult {
            absorbed: true,
            similarity_to_existing: similarity,
            message: format!(
                "Absorbed cognitive update for {:?} with phi={:.3}",
                update.category, update.phi
            ),
        }
    }
}

impl Default for KnowledgeManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graduation_high_quality_graduates() {
        let km = KnowledgeManager::new();
        let decision = km.evaluate_for_graduation(0.7, 5, 5);
        assert!(matches!(decision, GraduationDecision::Graduate));
    }

    #[test]
    fn graduation_low_quality_rejects() {
        let km = KnowledgeManager::new();
        let decision = km.evaluate_for_graduation(0.1, 1, 0);
        assert!(matches!(decision, GraduationDecision::Reject(_)));
    }

    #[test]
    fn graduation_medium_quality_defers() {
        let km = KnowledgeManager::new();
        let decision = km.evaluate_for_graduation(0.4, 3, 1);
        assert!(matches!(decision, GraduationDecision::Defer(_)));
    }

    #[test]
    fn search_returns_results_sorted_by_phi() {
        let mut km = KnowledgeManager::new();
        km.add_article(
            "1".to_string(),
            "DNS Troubleshooting".to_string(),
            SupportCategory::Network,
            0.3,
        );
        km.add_article(
            "2".to_string(),
            "DNS Configuration".to_string(),
            SupportCategory::Network,
            0.8,
        );
        km.add_article(
            "3".to_string(),
            "DNS Caching".to_string(),
            SupportCategory::Network,
            0.5,
        );

        let results = km.search("DNS", 10);
        assert_eq!(results.len(), 3);
        assert!((results[0].phi - 0.8).abs() < f64::EPSILON);
        assert!((results[1].phi - 0.5).abs() < f64::EPSILON);
        assert!((results[2].phi - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn deprecated_articles_excluded_from_search() {
        let mut km = KnowledgeManager::new();
        km.add_article(
            "1".to_string(),
            "Old DNS Guide".to_string(),
            SupportCategory::Network,
            0.5,
        );
        km.add_article(
            "2".to_string(),
            "New DNS Guide".to_string(),
            SupportCategory::Network,
            0.7,
        );
        km.deprecate("1");

        let results = km.search("DNS", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].article_id, "2");
    }

    #[test]
    fn search_filtered_excludes_local_only_when_sharing() {
        let mut km = KnowledgeManager::new();
        km.add_article(
            "1".to_string(),
            "DNS Public Guide".to_string(),
            SupportCategory::Network,
            0.7,
        );
        km.add_article_local_only(
            "2".to_string(),
            "DNS Local Notes".to_string(),
            SupportCategory::Network,
            0.8,
        );

        // When can_share=true, local-only articles are excluded
        let shared = km.search_filtered("DNS", 10, true);
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].article_id, "1");

        // When can_share=false, all articles appear
        let local = km.search_filtered("DNS", 10, false);
        assert_eq!(local.len(), 2);
    }

    #[test]
    fn absorption_works() {
        let mut km = KnowledgeManager::new();
        let update = CognitiveUpdateData {
            category: SupportCategory::Network,
            encoding: vec![1, 2, 3],
            phi: 0.6,
            resolution_pattern: "Restart DNS service".to_string(),
        };
        let result = km.absorb_cognitive_update(&update);
        assert!(result.absorbed);
        assert!(result.message.contains("Network"));
    }

    #[test]
    fn absorption_detects_existing_category() {
        let mut km = KnowledgeManager::new();
        km.add_article(
            "1".to_string(),
            "Existing".to_string(),
            SupportCategory::Network,
            0.5,
        );

        let update = CognitiveUpdateData {
            category: SupportCategory::Network,
            encoding: vec![1, 2, 3],
            phi: 0.6,
            resolution_pattern: "Restart DNS service".to_string(),
        };
        let result = km.absorb_cognitive_update(&update);
        assert!(result.absorbed);
        assert!((result.similarity_to_existing - 0.5).abs() < f32::EPSILON);
    }
}
