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
    LocalReasoning,
    /// Review may produce a promotion candidate, but never publication authority.
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

/// Retrieval never mints a `Federatable` state. Publication requires a distinct
/// promotion/authorization transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnowledgeShareabilityV1 {
    LocalOnly,
    ReviewRequired,
    NotEstablished,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportKnowledgeHitV1 {
    pub source_id: String,
    pub title: String,
    /// Retrieval relevance [0,1].
    pub similarity: f32,
    /// Epistemic confidence only when the provider actually establishes one.
    pub confidence: Option<f32>,
    /// Provider-specific quality signal (for example support-resolution phi).
    /// It is deliberately not relabeled as factual confidence.
    pub provider_quality: Option<f32>,
    pub category: Option<SupportCategory>,
    pub origin: KnowledgeOriginV1,
    pub shareability: KnowledgeShareabilityV1,
    /// `None` means technology applicability is not established.
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
        if let Some(confidence) = self.confidence {
            validate_unit(confidence, "confidence")?;
        }
        if let Some(quality) = self.provider_quality {
            validate_unit(quality, "provider quality")?;
        }
        if let Some(technology) = &self.technology {
            technology
                .validate()
                .map_err(|err| KnowledgeSourceErrorV1::InvalidTechnology(err.to_string()))?;
        }
        Ok(())
    }
}

pub trait SupportKnowledgeSourceV1 {
    fn search_support_knowledge(
        &mut self,
        query: &SupportKnowledgeQueryV1,
    ) -> Result<Vec<SupportKnowledgeHitV1>, KnowledgeSourceErrorV1>;
}

/// Merge heterogeneous hits without collapsing confidence, relevance, quality,
/// origin, or shareability into one scalar. Relevance is primary; established
/// confidence is only a tie-breaker.
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
            let replace = match by_source.get(&hit.source_id) {
                None => true,
                Some(existing) => rank_key(&hit) > rank_key(existing),
            };
            if replace {
                by_source.insert(hit.source_id.clone(), hit);
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
                    .unwrap_or(-1.0)
                    .partial_cmp(&a.confidence.unwrap_or(-1.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.source_id.cmp(&b.source_id))
    });
    merged.truncate(limit);
    Ok(merged)
}

fn rank_key(hit: &SupportKnowledgeHitV1) -> (u32, u32) {
    (hit.similarity.to_bits(), hit.confidence.unwrap_or(-1.0).to_bits())
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
            Self::InvalidUnitValue { label, value } => write!(f, "invalid knowledge {label} value {value}"),
            Self::InvalidTechnology(message) => write!(f, "invalid technology identity: {message}"),
            Self::Provider(message) => write!(f, "knowledge provider error: {message}"),
        }
    }
}
impl Error for KnowledgeSourceErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: &str, similarity: f32, confidence: Option<f32>) -> SupportKnowledgeHitV1 {
        SupportKnowledgeHitV1 {
            source_id: id.into(),
            title: format!("hit {id}"),
            similarity,
            confidence,
            provider_quality: None,
            category: None,
            origin: KnowledgeOriginV1::GlobalSemanticGraph,
            shareability: KnowledgeShareabilityV1::NotEstablished,
            technology: None,
        }
    }

    #[test]
    fn query_requires_text_and_nonzero_limit() {
        let empty = SupportKnowledgeQueryV1 {
            text: " ".into(), limit: 5, category: None, technology: None,
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        };
        assert!(matches!(empty.validate(), Err(KnowledgeSourceErrorV1::EmptyQuery)));
        let zero = SupportKnowledgeQueryV1 {
            text: "dns".into(), limit: 0, category: None, technology: None,
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        };
        assert!(matches!(zero.validate(), Err(KnowledgeSourceErrorV1::ZeroLimit)));
    }

    #[test]
    fn merge_deduplicates_and_ranks_relevance_first() {
        let merged = merge_knowledge_hits_v1(
            [
                vec![hit("a", 0.7, Some(0.9)), hit("b", 0.9, Some(0.5))],
                vec![hit("a", 0.8, Some(0.8)), hit("c", 0.6, None)],
            ], 3,
        ).unwrap();
        let ids: Vec<&str> = merged.iter().map(|item| item.source_id.as_str()).collect();
        assert_eq!(ids, vec!["b", "a", "c"]);
    }

    #[test]
    fn unknown_confidence_is_valid_and_remains_unknown() {
        let item = hit("a", 0.7, None);
        item.validate().unwrap();
        assert!(item.confidence.is_none());
    }

    #[test]
    fn retrieval_has_no_federatable_state() {
        for state in [KnowledgeShareabilityV1::LocalOnly, KnowledgeShareabilityV1::ReviewRequired, KnowledgeShareabilityV1::NotEstablished] {
            assert_ne!(format!("{state:?}"), "Federatable");
        }
    }
}
