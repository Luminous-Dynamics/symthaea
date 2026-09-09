// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Read-through adapter from Symthaea's global semantic KnowledgeManager into
//! the provider-neutral IT support knowledge-source contract.
//!
//! This bridge is intentionally retrieval-only:
//!
//! ```text
//! retrieved global fact != support article
//! retrieved global fact != technology applicability
//! retrieved global fact != sharing permission
//! retrieved global fact != standards authority
//! retrieved global fact != promotion authority
//! ```

use super::{KnowledgeEncoder, KnowledgeManager};
use symthaea_support::types::SupportCategory;
use symthaea_support::{
    KnowledgeAuthorityClassV1, KnowledgeLifecycleV1, KnowledgeOriginV1,
    KnowledgeShareabilityV1, KnowledgeSourceErrorV1, KnowledgeStabilityV1,
    SupportKnowledgeHitV1, SupportKnowledgeQueryV1, SupportKnowledgeSourceV1,
};

pub struct GlobalKnowledgeSupportAdapterV1<'a> {
    manager: &'a KnowledgeManager,
    encoder: KnowledgeEncoder,
}

impl<'a> GlobalKnowledgeSupportAdapterV1<'a> {
    pub fn new(manager: &'a KnowledgeManager) -> Self {
        Self {
            manager,
            encoder: KnowledgeEncoder::new(),
        }
    }
}

impl SupportKnowledgeSourceV1 for GlobalKnowledgeSupportAdapterV1<'_> {
    fn search_support_knowledge(
        &mut self,
        query: &SupportKnowledgeQueryV1,
    ) -> Result<Vec<SupportKnowledgeHitV1>, KnowledgeSourceErrorV1> {
        query.validate()?;
        let query_hv = self.encoder.encode_token(query.text.trim());
        let mut hits = Vec::new();

        for fact in self.manager.graph().all_facts() {
            let category = fact.domain.as_deref().and_then(map_support_category);
            if let Some(required) = &query.category {
                if category.as_ref() != Some(required) {
                    continue;
                }
            }

            let similarity = fact.encoding.vector.similarity(&query_hv);
            if !similarity.is_finite() || similarity <= 0.0 {
                continue;
            }

            hits.push(SupportKnowledgeHitV1 {
                source_id: format!("global-fact:{}", fact.id),
                title: fact.encoding.source_text.clone(),
                similarity: similarity.clamp(0.0, 1.0),
                confidence: if fact.confidence.is_finite() {
                    Some(fact.confidence.clamp(0.0, 1.0))
                } else {
                    None
                },
                provider_quality: None,
                category,
                origin: KnowledgeOriginV1::GlobalSemanticGraph,
                lifecycle: KnowledgeLifecycleV1::Unknown,
                authority: KnowledgeAuthorityClassV1::Unknown,
                stability: KnowledgeStabilityV1::Unspecified,
                shareability: KnowledgeShareabilityV1::NotEstablished,
                technology: None,
            });
        }

        hits.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.confidence
                        .unwrap_or(-1.0)
                        .partial_cmp(&a.confidence.unwrap_or(-1.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.source_id.cmp(&b.source_id))
        });
        hits.truncate(query.limit);
        Ok(hits)
    }
}

fn map_support_category(domain: &str) -> Option<SupportCategory> {
    match domain.trim().to_ascii_lowercase().as_str() {
        "network" | "networking" => Some(SupportCategory::Network),
        "hardware" | "computer-hardware" => Some(SupportCategory::Hardware),
        "software" | "computing" | "information-technology" => Some(SupportCategory::Software),
        "security" | "cybersecurity" | "information-security" => Some(SupportCategory::Security),
        "holochain" => Some(SupportCategory::Holochain),
        "mycelix" => Some(SupportCategory::Mycelix),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_support::{KnowledgeQueryPurposeV1, SupportKnowledgeQueryV1};

    #[test]
    fn lookup_is_read_through_and_does_not_mint_authority() {
        let mut manager = KnowledgeManager::default();
        manager.process("DNS failures can interrupt name resolution.", 1);
        let before = manager.graph().len();

        let mut adapter = GlobalKnowledgeSupportAdapterV1::new(&manager);
        let hits = adapter
            .search_support_knowledge(&SupportKnowledgeQueryV1 {
                text: "DNS failures".into(),
                limit: 5,
                category: None,
                technology: None,
                purpose: KnowledgeQueryPurposeV1::LocalReasoning,
            })
            .unwrap();

        assert_eq!(manager.graph().len(), before);
        assert!(hits.iter().all(|hit| {
            hit.origin == KnowledgeOriginV1::GlobalSemanticGraph
                && hit.lifecycle == KnowledgeLifecycleV1::Unknown
                && hit.authority == KnowledgeAuthorityClassV1::Unknown
                && hit.shareability == KnowledgeShareabilityV1::NotEstablished
                && hit.technology.is_none()
        }));
    }

    #[test]
    fn technology_query_context_is_not_copied_into_global_fact() {
        let mut manager = KnowledgeManager::default();
        manager.process("PostgreSQL uses transactions.", 1);
        let technology = symthaea_support::TechnologyIdentityV1 {
            ecosystem: Some("postgresql".into()),
            vendor: None,
            product: "postgresql".into(),
            edition: None,
            version: Some("18".into()),
            build: None,
            architecture: None,
            platform: Some("linux".into()),
            profile: None,
            observed_features: Default::default(),
        };

        let mut adapter = GlobalKnowledgeSupportAdapterV1::new(&manager);
        let hits = adapter
            .search_support_knowledge(&SupportKnowledgeQueryV1 {
                text: "PostgreSQL transactions".into(),
                limit: 5,
                category: None,
                technology: Some(technology),
                purpose: KnowledgeQueryPurposeV1::LocalReasoning,
            })
            .unwrap();
        assert!(hits.iter().all(|hit| hit.technology.is_none()));
    }

    #[test]
    fn unknown_global_domain_does_not_satisfy_category_constraint() {
        assert_eq!(map_support_category("geopolitics"), None);
        assert_eq!(
            map_support_category("networking"),
            Some(SupportCategory::Network)
        );
    }
}
