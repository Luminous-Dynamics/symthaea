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
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
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

/// Stable namespace for one knowledge provider.
///
/// This is an identity coordinate only. It does not establish provider authenticity,
/// independence, trustworthiness, or authority.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct KnowledgeProviderIdV1(String);

impl KnowledgeProviderIdV1 {
    pub fn try_new(value: impl Into<String>) -> Result<Self, KnowledgeSourceErrorV1> {
        let value = value.into();
        validate_identity_component(&value, "provider_id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for KnowledgeProviderIdV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::try_new(value).map_err(D::Error::custom)
    }
}

impl fmt::Display for KnowledgeProviderIdV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Provider-local stable record identity.
///
/// This identity is meaningful only when paired with a `KnowledgeProviderIdV1`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct KnowledgeRecordIdV1(String);

impl KnowledgeRecordIdV1 {
    pub fn try_new(value: impl Into<String>) -> Result<Self, KnowledgeSourceErrorV1> {
        let value = value.into();
        validate_identity_component(&value, "source_id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for KnowledgeRecordIdV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::try_new(value).map_err(D::Error::custom)
    }
}

impl fmt::Display for KnowledgeRecordIdV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportKnowledgeHitV1 {
    /// Provider namespace for this retrieval record. Identity only; not trust.
    pub provider_id: KnowledgeProviderIdV1,
    /// Stable identity within `provider_id`.
    pub source_id: KnowledgeRecordIdV1,
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
/// origin, applicability, or shareability into one scalar.
///
/// Identity is `provider_id + source_id`. Bare provider-local record IDs are never
/// deduplicated across providers. When the *same* provider record is repeated,
/// record/authority metadata must agree exactly before ranking may choose between
/// duplicate retrieval observations.
pub fn merge_knowledge_hits_v1(
    sources: impl IntoIterator<Item = Vec<SupportKnowledgeHitV1>>,
    limit: usize,
) -> Result<Vec<SupportKnowledgeHitV1>, KnowledgeSourceErrorV1> {
    if limit == 0 {
        return Err(KnowledgeSourceErrorV1::ZeroLimit);
    }

    let mut by_source = std::collections::BTreeMap::<
        (KnowledgeProviderIdV1, KnowledgeRecordIdV1),
        SupportKnowledgeHitV1,
    >::new();
    for hits in sources {
        for hit in hits {
            hit.validate()?;
            let key = (hit.provider_id.clone(), hit.source_id.clone());
            let replace = match by_source.get(&key) {
                None => true,
                Some(existing) => {
                    require_compatible_identity_metadata(existing, &hit)?;
                    prefer_new_hit(&hit, existing)
                }
            };
            if replace {
                by_source.insert(key, hit);
            }
        }
    }

    let mut merged: Vec<_> = by_source.into_values().collect();
    merged.sort_by(compare_hits);
    merged.truncate(limit);
    Ok(merged)
}

fn require_compatible_identity_metadata(
    existing: &SupportKnowledgeHitV1,
    candidate: &SupportKnowledgeHitV1,
) -> Result<(), KnowledgeSourceErrorV1> {
    if existing.title != candidate.title
        || existing.origin != candidate.origin
        || existing.shareability != candidate.shareability
        || existing.category != candidate.category
        || existing.technology != candidate.technology
    {
        return Err(KnowledgeSourceErrorV1::SourceMetadataConflict {
            provider_id: existing.provider_id.clone(),
            source_id: existing.source_id.clone(),
        });
    }
    Ok(())
}

fn prefer_new_hit(candidate: &SupportKnowledgeHitV1, existing: &SupportKnowledgeHitV1) -> bool {
    compare_hits(candidate, existing) == std::cmp::Ordering::Less
}

/// Ordering suitable for `sort_by`: strongest hit sorts first.
fn compare_hits(a: &SupportKnowledgeHitV1, b: &SupportKnowledgeHitV1) -> std::cmp::Ordering {
    b.similarity
        .partial_cmp(&a.similarity)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| {
            b.confidence
                .unwrap_or(-1.0)
                .partial_cmp(&a.confidence.unwrap_or(-1.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| a.provider_id.cmp(&b.provider_id))
        .then_with(|| a.source_id.cmp(&b.source_id))
}

fn validate_identity_component(
    value: &str,
    label: &'static str,
) -> Result<(), KnowledgeSourceErrorV1> {
    if value.trim().is_empty() {
        return Err(KnowledgeSourceErrorV1::EmptyIdentityComponent(label));
    }
    if value.trim() != value {
        return Err(KnowledgeSourceErrorV1::NonCanonicalIdentityComponent(label));
    }
    Ok(())
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
    EmptyIdentityComponent(&'static str),
    NonCanonicalIdentityComponent(&'static str),
    EmptyTitle,
    InvalidUnitValue {
        label: &'static str,
        value: f32,
    },
    InvalidTechnology(String),
    SourceMetadataConflict {
        provider_id: KnowledgeProviderIdV1,
        source_id: KnowledgeRecordIdV1,
    },
    Provider(String),
}

impl fmt::Display for KnowledgeSourceErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyQuery => write!(f, "knowledge query is empty"),
            Self::ZeroLimit => write!(f, "knowledge query limit must be non-zero"),
            Self::EmptyIdentityComponent(label) => {
                write!(f, "knowledge hit {label} must not be empty")
            }
            Self::NonCanonicalIdentityComponent(label) => {
                write!(f, "knowledge hit {label} contains surrounding whitespace")
            }
            Self::EmptyTitle => write!(f, "knowledge hit title is empty"),
            Self::InvalidUnitValue { label, value } => {
                write!(f, "invalid knowledge {label} value {value}")
            }
            Self::InvalidTechnology(message) => {
                write!(f, "invalid technology identity: {message}")
            }
            Self::SourceMetadataConflict {
                provider_id,
                source_id,
            } => write!(
                f,
                "conflicting record/authority metadata for knowledge source {provider_id}:{source_id}"
            ),
            Self::Provider(message) => write!(f, "knowledge provider error: {message}"),
        }
    }
}
impl Error for KnowledgeSourceErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(
        provider: &str,
        id: &str,
        similarity: f32,
        confidence: Option<f32>,
    ) -> SupportKnowledgeHitV1 {
        SupportKnowledgeHitV1 {
            provider_id: KnowledgeProviderIdV1::try_new(provider).unwrap(),
            source_id: KnowledgeRecordIdV1::try_new(id).unwrap(),
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
            text: " ".into(),
            limit: 5,
            category: None,
            technology: None,
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        };
        assert!(matches!(
            empty.validate(),
            Err(KnowledgeSourceErrorV1::EmptyQuery)
        ));
        let zero = SupportKnowledgeQueryV1 {
            text: "dns".into(),
            limit: 0,
            category: None,
            technology: None,
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        };
        assert!(matches!(
            zero.validate(),
            Err(KnowledgeSourceErrorV1::ZeroLimit)
        ));
    }

    #[test]
    fn merge_deduplicates_same_provider_record_and_ranks_relevance_first() {
        let merged = merge_knowledge_hits_v1(
            [
                vec![
                    hit("provider-a", "a", 0.7, Some(0.9)),
                    hit("provider-a", "b", 0.9, Some(0.5)),
                ],
                vec![
                    hit("provider-a", "a", 0.8, Some(0.8)),
                    hit("provider-a", "c", 0.6, None),
                ],
            ],
            3,
        )
        .unwrap();
        let ids: Vec<&str> = merged
            .iter()
            .map(|item| item.source_id.as_str())
            .collect();
        assert_eq!(ids, vec!["b", "a", "c"]);
    }

    #[test]
    fn same_local_source_id_from_distinct_providers_remains_distinct() {
        let merged = merge_knowledge_hits_v1(
            [
                vec![hit("provider-a", "same", 0.8, None)],
                vec![hit("provider-b", "same", 0.7, None)],
            ],
            10,
        )
        .unwrap();
        assert_eq!(merged.len(), 2);
        assert_ne!(merged[0].provider_id, merged[1].provider_id);
    }

    #[test]
    fn conflicting_shareability_for_same_source_identity_fails_closed() {
        let first = hit("provider-a", "a", 0.8, None);
        let mut conflicting = first.clone();
        conflicting.shareability = KnowledgeShareabilityV1::ReviewRequired;

        assert!(matches!(
            merge_knowledge_hits_v1([vec![first], vec![conflicting]], 10),
            Err(KnowledgeSourceErrorV1::SourceMetadataConflict { .. })
        ));
    }

    #[test]
    fn conflicting_title_for_same_source_identity_fails_closed() {
        let first = hit("provider-a", "a", 0.8, None);
        let mut conflicting = first.clone();
        conflicting.title = "different record semantics".into();

        assert!(matches!(
            merge_knowledge_hits_v1([vec![first], vec![conflicting]], 10),
            Err(KnowledgeSourceErrorV1::SourceMetadataConflict { .. })
        ));
    }

    #[test]
    fn established_confidence_beats_unknown_only_within_same_source_identity() {
        let merged = merge_knowledge_hits_v1(
            [
                vec![hit("provider-a", "a", 0.8, None)],
                vec![hit("provider-a", "a", 0.8, Some(0.2))],
            ],
            1,
        )
        .unwrap();
        assert_eq!(merged[0].confidence, Some(0.2));
    }

    #[test]
    fn provider_identity_is_required_canonical_and_serde_safe() {
        assert!(matches!(
            KnowledgeProviderIdV1::try_new(" "),
            Err(KnowledgeSourceErrorV1::EmptyIdentityComponent("provider_id"))
        ));
        assert!(matches!(
            KnowledgeProviderIdV1::try_new(" provider-a "),
            Err(KnowledgeSourceErrorV1::NonCanonicalIdentityComponent(
                "provider_id"
            ))
        ));
        assert!(serde_json::from_str::<KnowledgeProviderIdV1>("\" provider-a \"").is_err());
        assert_eq!(
            serde_json::from_str::<KnowledgeProviderIdV1>("\"provider-a\"")
                .unwrap()
                .as_str(),
            "provider-a"
        );
    }

    #[test]
    fn record_identity_is_required_canonical_and_serde_safe() {
        assert!(matches!(
            KnowledgeRecordIdV1::try_new(" "),
            Err(KnowledgeSourceErrorV1::EmptyIdentityComponent("source_id"))
        ));
        assert!(matches!(
            KnowledgeRecordIdV1::try_new(" record-a "),
            Err(KnowledgeSourceErrorV1::NonCanonicalIdentityComponent(
                "source_id"
            ))
        ));
        assert!(serde_json::from_str::<KnowledgeRecordIdV1>("\" record-a \"").is_err());
    }

    #[test]
    fn unknown_confidence_is_valid_and_remains_unknown() {
        let item = hit("provider-a", "a", 0.7, None);
        item.validate().unwrap();
        assert!(item.confidence.is_none());
    }

    #[test]
    fn retrieval_has_no_federatable_state() {
        for state in [
            KnowledgeShareabilityV1::LocalOnly,
            KnowledgeShareabilityV1::ReviewRequired,
            KnowledgeShareabilityV1::NotEstablished,
        ] {
            assert_ne!(format!("{state:?}"), "Federatable");
        }
    }
}
