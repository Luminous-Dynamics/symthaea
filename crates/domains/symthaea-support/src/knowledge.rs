// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Local support knowledge store — article management, graduation scoring, and
//! cognitive-update absorption.
//!
//! This is a support-domain store, not Symthaea's global semantic KnowledgeManager.
//! It implements `SupportKnowledgeSourceV1` so callers can merge local articles
//! with richer providers without merging storage or authority semantics.

use crate::knowledge_source::{
    KnowledgeOriginV1, KnowledgeQueryPurposeV1, KnowledgeShareabilityV1,
    KnowledgeSourceErrorV1, SupportKnowledgeHitV1, SupportKnowledgeQueryV1,
    SupportKnowledgeSourceV1,
};
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
    /// Support-quality signal; not factual confidence.
    phi: f64,
    consolidation_strength: f64,
    #[allow(dead_code)]
    retrieval_count: u32,
    deprecated: bool,
    local_only: bool,
}

impl KnowledgeManager {
    pub fn new() -> Self {
        Self { articles: Vec::new() }
    }

    /// Evaluate whether a resolution is strong enough to become a *promotion
    /// candidate*. This does not establish universal truth or federation authority.
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

    /// Legacy local keyword search. Prefer `SupportKnowledgeSourceV1` for new
    /// cross-provider support reasoning.
    pub fn search(&self, query: &str, k: usize) -> Vec<KnowledgeMatch> {
        self.search_filtered(query, k, false)
    }

    /// Legacy local search with federation-oriented local-only filtering.
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

    pub fn deprecate(&mut self, article_id: &str) -> bool {
        if let Some(article) = self.articles.iter_mut().find(|a| a.id == article_id) {
            article.deprecated = true;
            article.consolidation_strength *= 0.1;
            true
        } else {
            false
        }
    }

    /// Absorb a cognitive update into the support-domain compatibility store.
    /// `absorbed=true` means accepted by this local mechanism, not validated as a
    /// globally applicable technical fact.
    pub fn absorb_cognitive_update(&mut self, update: &CognitiveUpdateData) -> AbsorptionResult {
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

impl SupportKnowledgeSourceV1 for KnowledgeManager {
    fn search_support_knowledge(
        &mut self,
        query: &SupportKnowledgeQueryV1,
    ) -> Result<Vec<SupportKnowledgeHitV1>, KnowledgeSourceErrorV1> {
        query.validate()?;
        let needle = query.text.trim().to_lowercase();
        let mut hits = Vec::new();

        for article in &self.articles {
            if article.deprecated {
                continue;
            }
            if matches!(query.purpose, KnowledgeQueryPurposeV1::PromotionReview)
                && article.local_only
            {
                continue;
            }
            if let Some(category) = &query.category {
                if &article.category != category {
                    continue;
                }
            }
            let title = article.title.to_lowercase();
            if !title.contains(&needle) {
                continue;
            }

            let provider_quality = if article.phi.is_finite() && (0.0..=1.0).contains(&article.phi) {
                Some(article.phi as f32)
            } else {
                None
            };
            let similarity = if article.consolidation_strength.is_finite() {
                article.consolidation_strength.clamp(0.0, 1.0) as f32
            } else {
                0.0
            };
            hits.push(SupportKnowledgeHitV1 {
                source_id: format!("support-article:{}", article.id),
                title: article.title.clone(),
                similarity,
                confidence: None,
                provider_quality,
                category: Some(article.category.clone()),
                origin: KnowledgeOriginV1::SupportArticle,
                shareability: if article.local_only {
                    KnowledgeShareabilityV1::LocalOnly
                } else {
                    KnowledgeShareabilityV1::ReviewRequired
                },
                // Existing articles do not yet carry exact technology identity.
                technology: None,
            });
        }

        hits.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.provider_quality
                        .unwrap_or(-1.0)
                        .partial_cmp(&a.provider_quality.unwrap_or(-1.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.source_id.cmp(&b.source_id))
        });
        hits.truncate(query.limit);
        Ok(hits)
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
    fn graduation_high_quality_becomes_candidate_quality() {
        let km = KnowledgeManager::new();
        assert!(matches!(km.evaluate_for_graduation(0.7, 5, 5), GraduationDecision::Graduate));
    }

    #[test]
    fn graduation_low_quality_rejects() {
        let km = KnowledgeManager::new();
        assert!(matches!(km.evaluate_for_graduation(0.1, 1, 0), GraduationDecision::Reject(_)));
    }

    #[test]
    fn graduation_medium_quality_defers() {
        let km = KnowledgeManager::new();
        assert!(matches!(km.evaluate_for_graduation(0.4, 3, 1), GraduationDecision::Defer(_)));
    }

    #[test]
    fn legacy_search_returns_results_sorted_by_phi() {
        let mut km = KnowledgeManager::new();
        km.add_article("1".into(), "DNS Troubleshooting".into(), SupportCategory::Network, 0.3);
        km.add_article("2".into(), "DNS Configuration".into(), SupportCategory::Network, 0.8);
        km.add_article("3".into(), "DNS Caching".into(), SupportCategory::Network, 0.5);
        let results = km.search("DNS", 10);
        assert_eq!(results.len(), 3);
        assert!((results[0].phi - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn promotion_review_excludes_local_only_but_local_reasoning_can_use_it() {
        let mut km = KnowledgeManager::new();
        km.add_article("1".into(), "DNS Public Guide".into(), SupportCategory::Network, 0.7);
        km.add_article_local_only("2".into(), "DNS Local Notes".into(), SupportCategory::Network, 0.8);

        let mut query = SupportKnowledgeQueryV1 {
            text: "DNS".into(),
            limit: 10,
            category: Some(SupportCategory::Network),
            technology: None,
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        };
        let local = km.search_support_knowledge(&query).unwrap();
        assert_eq!(local.len(), 2);
        assert!(local.iter().any(|h| h.shareability == KnowledgeShareabilityV1::LocalOnly));

        query.purpose = KnowledgeQueryPurposeV1::PromotionReview;
        let review = km.search_support_knowledge(&query).unwrap();
        assert_eq!(review.len(), 1);
        assert_eq!(review[0].shareability, KnowledgeShareabilityV1::ReviewRequired);
    }

    #[test]
    fn support_phi_is_quality_not_factual_confidence() {
        let mut km = KnowledgeManager::new();
        km.add_article("1".into(), "DNS Guide".into(), SupportCategory::Network, 0.8);
        let query = SupportKnowledgeQueryV1 {
            text: "DNS".into(), limit: 5, category: None, technology: None,
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        };
        let hit = km.search_support_knowledge(&query).unwrap().remove(0);
        assert_eq!(hit.confidence, None);
        assert_eq!(hit.provider_quality, Some(0.8));
    }

    #[test]
    fn deprecated_articles_excluded() {
        let mut km = KnowledgeManager::new();
        km.add_article("1".into(), "Old DNS Guide".into(), SupportCategory::Network, 0.5);
        km.deprecate("1");
        assert!(km.search("DNS", 10).is_empty());
    }

    #[test]
    fn absorption_detects_existing_category() {
        let mut km = KnowledgeManager::new();
        km.add_article("1".into(), "Existing".into(), SupportCategory::Network, 0.5);
        let update = CognitiveUpdateData {
            category: SupportCategory::Network,
            encoding: vec![1, 2, 3],
            phi: 0.6,
            resolution_pattern: "Restart DNS service".into(),
        };
        let result = km.absorb_cognitive_update(&update);
        assert!(result.absorbed);
        assert!((result.similarity_to_existing - 0.5).abs() < f32::EPSILON);
    }
}
