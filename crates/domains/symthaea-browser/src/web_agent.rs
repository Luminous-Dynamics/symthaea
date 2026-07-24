// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Prism-first web research orchestration.
//!
//! This layer deliberately distinguishes local knowledge, search-result
//! snippets, retrieved-page evidence, and corroborated claims. Search snippets
//! are discovery candidates only and are never represented as verified facts or
//! made eligible for durable Prism ingestion.

use std::collections::BTreeSet;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::actions::BrowserAction;
use crate::cdp::CdpSession;
use crate::executor::BrowserExecutor;
use crate::observation::PageObservation;
use crate::safety::BrowserSafetyPolicy;

/// Epistemic status of a claim-like text fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceStatus {
    /// Existing local knowledge supplied by Prism or another trusted store.
    LocalKnowledge,
    /// Text observed on a search-result page. Discovery only; not verification.
    SearchResultSnippet,
    /// Text extracted from the actual cited page but not independently checked.
    RetrievedPage,
    /// Claim supported by independent evidence and contradiction checks.
    Corroborated,
}

/// A claim candidate with explicit epistemic status and source provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebClaim {
    pub content: String,
    /// Canonical page from which this text was observed.
    pub source: String,
    /// Calibrated confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    pub evidence_status: EvidenceStatus,
    /// Exact bounded evidence text supporting this candidate.
    pub evidence_excerpt: Option<String>,
    /// Search engine, local index, or retrieval mechanism that surfaced it.
    pub retrieved_from: Option<String>,
}

impl WebClaim {
    /// Construct a local claim supplied by an upstream knowledge system.
    pub fn local(content: impl Into<String>, source: impl Into<String>, confidence: f32) -> Self {
        Self {
            content: content.into(),
            source: source.into(),
            confidence: confidence.clamp(0.0, 1.0),
            evidence_status: EvidenceStatus::LocalKnowledge,
            evidence_excerpt: None,
            retrieved_from: Some("Prism".to_string()),
        }
    }

    /// Whether this claim may enter a durable knowledge store without another
    /// promotion step.
    pub fn eligible_for_prism_ingest(&self) -> bool {
        matches!(self.evidence_status, EvidenceStatus::Corroborated)
            && self.confidence.is_finite()
            && self.confidence >= 0.75
            && self.evidence_excerpt.is_some()
    }
}

/// Result of a browser research pass.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WebResearchResult {
    pub claims: Vec<WebClaim>,
    /// Actual successfully observed final URLs, not configured engine bases.
    pub urls_visited: Vec<String>,
}

/// Result of the Prism-first query pipeline.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WebAgentResult {
    pub local_claims: Vec<WebClaim>,
    pub web_claims: Vec<WebClaim>,
    pub urls_visited: Vec<String>,
    pub answered_locally: bool,
}

const MIN_SUFFICIENT_LOCAL_CLAIMS: usize = 2;
const MAX_SEARCH_CANDIDATES: usize = 12;

const SEARCH_URLS: &[(&str, &str)] = &[
    ("Wikipedia", "https://en.wikipedia.org/w/index.php?search="),
    ("DuckDuckGo", "https://html.duckduckgo.com/html/?q="),
];

/// The web agent: local knowledge first, browser discovery second.
pub struct WebAgent {
    safety: BrowserSafetyPolicy,
    phi: f64,
}

impl WebAgent {
    pub fn new(safety: BrowserSafetyPolicy, phi: f64) -> Self {
        Self { safety, phi }
    }

    pub fn set_phi(&mut self, phi: f64) {
        self.phi = phi;
    }

    pub fn can_navigate_url(&self, url: &str) -> bool {
        self.safety.is_action_allowed(
            &BrowserAction::Navigate {
                url: url.to_string(),
            },
            self.phi,
        )
    }

    pub fn can_navigate(&self) -> bool {
        self.can_navigate_url("https://example.com/")
    }

    /// Browse search-result pages and return low-confidence discovery
    /// candidates with actual visited URLs.
    pub async fn research_detailed(
        &self,
        session: &CdpSession,
        query: &str,
    ) -> Result<WebResearchResult> {
        let executor = BrowserExecutor::new(session, &self.safety, self.phi);
        let encoded_query: String =
            url::form_urlencoded::byte_serialize(query.as_bytes()).collect();
        let mut result = WebResearchResult::default();
        let mut seen_claims = BTreeSet::new();

        for &(engine_name, base_url) in SEARCH_URLS {
            let requested_url = format!("{base_url}{encoded_query}");
            let receipt = executor
                .execute(BrowserAction::Navigate {
                    url: requested_url.clone(),
                })
                .await;
            if !receipt.succeeded() {
                tracing::warn!(
                    engine = engine_name,
                    outcome = ?receipt.outcome,
                    "Search navigation denied or failed"
                );
                continue;
            }

            let observation = match session.observe().await {
                Ok(observation) => observation,
                Err(error) => {
                    tracing::warn!(
                        engine = engine_name,
                        error = %error,
                        "Search observation failed"
                    );
                    continue;
                }
            };

            let final_url = observation.redacted_url();
            if !result.urls_visited.contains(&final_url) {
                result.urls_visited.push(final_url.clone());
            }

            for candidate in extract_search_candidates(&observation, engine_name) {
                let normalized = normalize_claim_key(&candidate.content);
                if seen_claims.insert(normalized) {
                    result.claims.push(candidate);
                }
                if result.claims.len() >= MAX_SEARCH_CANDIDATES {
                    return Ok(result);
                }
            }
        }

        Ok(result)
    }

    /// Compatibility helper returning only the discovery candidates.
    pub async fn research(&self, session: &CdpSession, query: &str) -> Result<Vec<WebClaim>> {
        Ok(self.research_detailed(session, query).await?.claims)
    }

    /// Full query pipeline using actual local claims rather than a count that
    /// cannot be inspected for quality.
    pub async fn query_with_fallback(
        &self,
        session: &CdpSession,
        query: &str,
        local_claims: Vec<WebClaim>,
    ) -> Result<WebAgentResult> {
        if local_knowledge_is_sufficient(&local_claims) {
            return Ok(WebAgentResult {
                local_claims,
                web_claims: Vec::new(),
                urls_visited: Vec::new(),
                answered_locally: true,
            });
        }

        let web = self.research_detailed(session, query).await?;
        Ok(WebAgentResult {
            local_claims,
            web_claims: web.claims,
            urls_visited: web.urls_visited,
            answered_locally: false,
        })
    }
}

fn local_knowledge_is_sufficient(claims: &[WebClaim]) -> bool {
    if claims.len() < MIN_SUFFICIENT_LOCAL_CLAIMS {
        return false;
    }
    let trustworthy: Vec<&WebClaim> = claims
        .iter()
        .filter(|claim| {
            matches!(
                claim.evidence_status,
                EvidenceStatus::LocalKnowledge | EvidenceStatus::Corroborated
            ) && claim.confidence.is_finite()
                && claim.confidence >= 0.70
        })
        .collect();
    if trustworthy.len() < MIN_SUFFICIENT_LOCAL_CLAIMS {
        return false;
    }
    let mean = trustworthy
        .iter()
        .map(|claim| claim.confidence)
        .sum::<f32>()
        / trustworthy.len() as f32;
    mean >= 0.80
}

fn extract_search_candidates(observation: &PageObservation, engine_name: &str) -> Vec<WebClaim> {
    let source = observation.redacted_url();
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();

    for element in &observation.elements {
        if !matches!(
            element.role.to_ascii_lowercase().as_str(),
            "link" | "heading" | "text" | "statictext" | "paragraph" | "listitem"
        ) {
            continue;
        }

        for sentence in split_candidate_sentences(&element.name) {
            let key = normalize_claim_key(sentence);
            if seen.insert(key) {
                candidates.push(WebClaim {
                    content: sentence.to_string(),
                    source: source.clone(),
                    confidence: 0.20,
                    evidence_status: EvidenceStatus::SearchResultSnippet,
                    evidence_excerpt: Some(sentence.to_string()),
                    retrieved_from: Some(engine_name.to_string()),
                });
            }
            if candidates.len() >= MAX_SEARCH_CANDIDATES {
                return candidates;
            }
        }
    }
    candidates
}

fn split_candidate_sentences(text: &str) -> impl Iterator<Item = &str> {
    text.split(|character: char| matches!(character, '.' | '!' | '\n'))
        .map(str::trim)
        .filter(|sentence| sentence.len() > 40 && sentence.len() < 500)
        .filter(|sentence| !sentence.contains('?'))
        .filter(|sentence| {
            sentence
                .chars()
                .filter(|character| character.is_alphabetic())
                .count()
                > sentence.chars().count() / 2
        })
}

fn normalize_claim_key(claim: &str) -> String {
    claim
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::AccessibleElement;

    fn search_observation() -> PageObservation {
        PageObservation {
            url: "https://search.example/?q=consciousness".into(),
            title: "Search".into(),
            elements: vec![
                AccessibleElement {
                    backend_node_id: 1,
                    role: "link".into(),
                    name: "Consciousness is commonly described as awareness of internal and external existence in philosophical and scientific literature".into(),
                    value: None,
                    description: None,
                    focused: false,
                    disabled: false,
                },
                AccessibleElement {
                    backend_node_id: 2,
                    role: "button".into(),
                    name: "Search".into(),
                    value: None,
                    description: None,
                    focused: false,
                    disabled: false,
                },
            ],
            focused_element: None,
        }
    }

    #[test]
    fn snippets_are_discovery_only() {
        let claims = extract_search_candidates(&search_observation(), "ExampleSearch");
        assert_eq!(claims.len(), 1);
        assert_eq!(
            claims[0].evidence_status,
            EvidenceStatus::SearchResultSnippet
        );
        assert!(!claims[0].eligible_for_prism_ingest());
        assert_eq!(claims[0].confidence, 0.20);
    }

    #[test]
    fn local_sufficiency_uses_quality_not_count_alone() {
        let weak = vec![
            WebClaim::local("A", "local", 0.4),
            WebClaim::local("B", "local", 0.4),
            WebClaim::local("C", "local", 0.4),
        ];
        assert!(!local_knowledge_is_sufficient(&weak));

        let strong = vec![
            WebClaim::local("A", "local", 0.85),
            WebClaim::local("B", "local", 0.80),
        ];
        assert!(local_knowledge_is_sufficient(&strong));
    }

    #[test]
    fn only_corroborated_evidence_is_ingestible() {
        let mut claim = WebClaim {
            content: "A supported claim".into(),
            source: "https://example.com/evidence".into(),
            confidence: 0.9,
            evidence_status: EvidenceStatus::RetrievedPage,
            evidence_excerpt: Some("A supported claim".into()),
            retrieved_from: Some("direct".into()),
        };
        assert!(!claim.eligible_for_prism_ingest());
        claim.evidence_status = EvidenceStatus::Corroborated;
        assert!(claim.eligible_for_prism_ingest());
    }

    #[test]
    fn web_agent_phi_gating() {
        let safety = BrowserSafetyPolicy::default();
        let low_phi = WebAgent::new(safety.clone(), 0.1);
        assert!(!low_phi.can_navigate());

        let high_phi = WebAgent::new(safety, 0.5);
        assert!(high_phi.can_navigate());
    }
}
