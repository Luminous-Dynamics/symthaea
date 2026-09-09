// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Provider-neutral knowledge-source boundary for IT support.
//!
//! The support crate must not depend upward on the desktop/root KnowledgeManager.
//! Instead, knowledge providers implement this small read interface. Retrieval is
//! deliberately separate from federation/promotion authority.

use crate::technology::TechnologyIdentityV1;
use crate::types::SupportCategory;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnowledgeQueryPurposeV1 {
    /// Local reasoning/support. Local-only knowledge may be returned.
    LocalReasoning,
    /// Review workflow that may eventually produce a promotion candidate.
    /// This purpose still does not authorize publication/federation.
    PromotionReview,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportKnowledgeQueryV1 {
    pub text: String,
    pub limit: usize,
    pub category: Option<SupportCategory>,
    pub technology: Option<TechnologyIdentityV1>,
    pub purpose: KnowledgeQueryPurposeV1,
}

impl SupportKnowledgeQueryV1 {
    pub fn validate(&self) -> Result<(), KnowledgeSourceErrorV1> {
        if self.text.trim().is_empty() {
            return Err(KnowledgeSourceErrorV1::EmptyQuery);
        }
        if self.limit == 0 {
            return Err(KnowledgeSourceErrorV1::ZeroLimit);
        }
        if let Some(technology) = &self.technology {
            technology
                .validate()
                .map_err(|err| KnowledgeSourceErrorV1::InvalidTechnology(err.to_string()))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnowledgeOriginV1 {
    SupportArticle,
    GlobalSemanticGraph,
    ExternalStandard,
    VendorDocumentation,
    FederatedPeer,
    Other(String),
}

/// Shareability is intentionally conservative. Retrieval never mints a
/// `Federatable` state. Publication requires a distinct promotion transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnowledgeShareabilityV1 {
    /// Must not leave the local support context.
    LocalOnly,
    /// May be considered by a promotion/review process, but is not yet publishable.
    ReviewRequired,
    /// Provider does not establish sharing rights/semantics.
    NotEstablished,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportKnowledgeHitV1 {
    pub source_id: String,
    pub title: String,
    /// Retrieval relevance [0,1] where established by the provider.
    pub similarity: f32,
    /// Epistemic confidence [0,1] where established by the provider.
    pub confidence: f32,
    pub category: Option<SupportCategory>,
    pub origin: KnowledgeOriginV1,
    pub shareability: KnowledgeShareabilityV1,
    /// Optional exact technology context. `None` means applicability is not established.
    pub technology: Option<TechnologyIdentityV1>,
}

impl SupportKnowledgeHitV1 {
    pub fn validate(&self) -> Result<(), KnowledgeSourceErrorV1> {
        if self.source_id.trim().is_empty() {
            return Err(KnowledgeSourceErrorV1::EmptySourceId);
        }
        if self.title.trim().is_empty() {
            return Err(KnowledgeSourceErrorV1::EmptyTitle);
        }
        validate_unit(self.similarity, "similarity")?;
        validate_unit(self.confidence, "confidence")?;
        if let Some(technology) = &self.technology {
            technology
                .validate()
                .map_err(|err| KnowledgeSourceErrorV1::InvalidTechnology(err.to_string()))?;
        }
        Ok(())
    }
}

/// Small read interface implemented by local support storage, the desktop
/// KnowledgeManager adapter, and future standards/vendor providers.
pub trait SupportKnowledgeSourceV1 {
    fn search_support_knowledge(
        &mut self,
        query: &SupportKnowledgeQueryV1,
    ) -> Result<Vec<SupportKnowledgeHitV1>, KnowledgeSourceErrorV1>;
}

/// Merge heterogeneous provider hits without collapsing confidence, similarity,
/// origin, or shareability into one score. Ranking uses relevance first and
/// confidence second; exact duplicate source identities keep the strongest hit.
pub fn merge_knowledge_hits_v1(
    sources: impl IntoIterator<Item = Vec<SupportKnowledgeHitV1>>,
    limit: usize,
) -> Result<Vec<SupportKnowledgeHitV1>, KnowledgeSourceErrorV1> {
    if limit == 0 {
        return Err(KnowledgeSourceErrorV1::ZeroLimit);
    }

    let mut by_source = std::collections::BTreeMap::<String, SupportKnowledgeHitV1>::new();
    for hits in sources {
        for hit in hits {
            hit.validate()?;
            match by_source.get(&hit.source_id) {
                Some(existing)
                    if (existing.similarity, existing.confidence)
                        >= (hit.similarity, hit.confidence) => {}
                _ => {
                    by_source.insert(hit.source_id.clone(), hit);
                }
            }
        }
    }

    let mut merged: Vec<_> = by_source.into_values().collect();
    merged.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.source_id.cmp(&b.source_id))
    });
    merged.truncate(limit);
    Ok(merged)
}

fn validate_unit(value: f32, label: &'static str) -> Result<(), KnowledgeSourceErrorV1> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(KnowledgeSourceErrorV1::InvalidUnitValue { label, value })
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum KnowledgeSourceErrorV1 {
    EmptyQuery,
    ZeroLimit,
    EmptySourceId,
    EmptyTitle,
    InvalidUnitValue { label: &'static str, value: f32 },
    InvalidTechnology(String),
    Provider(String),
}

impl fmt::Display for KnowledgeSourceErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyQuery => write!(f, "knowledge query is empty"),
            Self::ZeroLimit => write!(f, "knowledge query limit must be non-zero"),
            Self::EmptySourceId => write!(f, "knowledge hit source id is empty"),
            Self::EmptyTitle => write!(f, "knowledge hit title is empty"),
            Self::InvalidUnitValue { label, value } => {
                write!(f, "invalid knowledge {label} value {value}")
            }
            Self::InvalidTechnology(message) => write!(f, "invalid technology identity: {message}"),
            Self::Provider(message) => write!(f, "knowledge provider error: {message}"),
        }
    }
}

impl Error for KnowledgeSourceErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: &str, similarity: f32, confidence: f32) -> SupportKnowledgeHitV1 {
        SupportKnowledgeHitV1 {
            source_id: id.into(),
            title: format!("hit {id}"),
            similarity,
            confidence,
            category: None,
            origin: KnowledgeOriginV1::GlobalSemanticGraph,
            shareability: KnowledgeShareabilityV1::NotEstablished,
            technology: None,
        }
    }

    #[test]
    fn query_requires_text_and_nonzero_limit() {
        let empty = SupportKnowledgeQueryV1 {
            text: " ".into(),
            limit: 5,
            category: None,
            technology: None,
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        };
        assert!(matches!(empty.validate(), Err(KnowledgeSourceErrorV1::EmptyQuery)));

        let zero = SupportKnowledgeQueryV1 {
            text: "dns".into(),
            limit: 0,
            category: None,
            technology: None,
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        };
        assert!(matches!(zero.validate(), Err(KnowledgeSourceErrorV1::ZeroLimit)));
    }

    #[test]
    fn merge_deduplicates_by_source_and_ranks_relevance_first() {
        let merged = merge_knowledge_hits_v1(
            [
                vec![hit("a", 0.7, 0.9), hit("b", 0.9, 0.5)],
                vec![hit("a", 0.8, 0.8), hit("c", 0.6, 1.0)],
            ],
            3,
        )
        .unwrap();
        let ids: Vec<&str> = merged.iter().map(|item| item.source_id.as_str()).collect();
        assert_eq!(ids, vec!["b", "a", "c"]);
        assert!((merged[1].similarity - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn no_retrieval_hit_is_implicitly_federatable() {
        for state in [
            KnowledgeShareabilityV1::LocalOnly,
            KnowledgeShareabilityV1::ReviewRequired,
            KnowledgeShareabilityV1::NotEstablished,
        ] {
            assert_ne!(format!("{state:?}"), "Federatable");
        }
    }
}
