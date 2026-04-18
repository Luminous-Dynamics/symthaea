// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Web agent: orchestrates Prism search + CDP browsing for grounded answers.
//!
//! Flow:
//! 1. User query → Prism local search (offline, sub-ms)
//! 2. If sufficient local results → return immediately
//! 3. If insufficient → browse web via CDP → extract content → verify claims
//! 4. Feed verified claims back into Prism knowledge base
//!
//! This is the integration layer between Prism (epistemic search) and
//! symthaea-browser (CDP-connected browser agent).

use crate::cdp::CdpSession;
use crate::observation::PageObservation;
use crate::safety::BrowserSafetyPolicy;
use crate::actions::BrowserAction;
use anyhow::Result;

/// Result of a web agent query — combines local + web sources.
#[derive(Debug, Clone)]
pub struct WebAgentResult {
    /// Claims found from local Prism search.
    pub local_claims: Vec<WebClaim>,
    /// Claims extracted from web browsing.
    pub web_claims: Vec<WebClaim>,
    /// URLs visited during research.
    pub urls_visited: Vec<String>,
    /// Whether the query was answered from local knowledge alone.
    pub answered_locally: bool,
}

/// A single verified claim with source information.
#[derive(Debug, Clone)]
pub struct WebClaim {
    pub content: String,
    pub source: String,
    pub confidence: f32,
}

/// Minimum local results before triggering web research.
const MIN_LOCAL_RESULTS: usize = 3;

/// Search URLs to try for web research.
const SEARCH_URLS: &[(&str, &str)] = &[
    ("Wikipedia", "https://en.wikipedia.org/w/index.php?search="),
    ("DuckDuckGo", "https://html.duckduckgo.com/html/?q="),
];

/// The web agent: Prism-first, browser-fallback.
pub struct WebAgent {
    safety: BrowserSafetyPolicy,
    phi: f64,
}

impl WebAgent {
    /// Create a web agent with the given safety policy and consciousness level.
    pub fn new(safety: BrowserSafetyPolicy, phi: f64) -> Self {
        Self { safety, phi }
    }

    /// Update the consciousness level (from cognitive loop).
    pub fn set_phi(&mut self, phi: f64) {
        self.phi = phi;
    }

    /// Check if the agent can navigate (Phi >= 0.30).
    pub fn can_navigate(&self) -> bool {
        self.safety.is_action_allowed(&BrowserAction::Navigate { url: String::new() }, self.phi)
    }

    /// Execute a web research session via CDP.
    ///
    /// Navigates to search engine, extracts results page, returns claims.
    /// Requires Phi >= 0.30 for navigation.
    pub async fn research(&self, session: &CdpSession, query: &str) -> Result<Vec<WebClaim>> {
        if !self.can_navigate() {
            tracing::info!(phi = self.phi, "Phi too low for web navigation");
            return Ok(vec![]);
        }

        let mut claims = Vec::new();
        let encoded_query = query.replace(' ', "+");

        for &(engine_name, base_url) in SEARCH_URLS {
            let url = format!("{}{}", base_url, encoded_query);

            tracing::info!(engine = engine_name, url = %url, "Web agent navigating");

            if let Err(e) = session.navigate(&url).await {
                tracing::warn!(engine = engine_name, error = %e, "Navigation failed");
                continue;
            }

            // Observe the page
            let obs = match session.observe().await {
                Ok(obs) => obs,
                Err(e) => {
                    tracing::warn!(engine = engine_name, error = %e, "Observation failed");
                    continue;
                }
            };

            // Extract text content from the page
            let page_text = obs.to_cognitive_text();

            // Extract claims from page text (simple sentence splitting)
            let page_claims = extract_claims_from_text(&page_text, &url);
            tracing::info!(
                engine = engine_name,
                claims = page_claims.len(),
                "Extracted claims from page"
            );

            claims.extend(page_claims);

            // One successful source is enough for basic queries
            if claims.len() >= MIN_LOCAL_RESULTS {
                break;
            }
        }

        Ok(claims)
    }

    /// Full query pipeline: Prism local → web fallback → merged results.
    ///
    /// `local_results` should come from `SearchEngine::search()`.
    pub async fn query_with_fallback(
        &self,
        session: &CdpSession,
        query: &str,
        local_result_count: usize,
    ) -> Result<WebAgentResult> {
        if local_result_count >= MIN_LOCAL_RESULTS {
            return Ok(WebAgentResult {
                local_claims: vec![],
                web_claims: vec![],
                urls_visited: vec![],
                answered_locally: true,
            });
        }

        let web_claims = self.research(session, query).await?;
        let urls: Vec<String> = SEARCH_URLS.iter().map(|(_, u)| u.to_string()).collect();

        Ok(WebAgentResult {
            local_claims: vec![],
            web_claims,
            urls_visited: urls,
            answered_locally: false,
        })
    }
}

/// Extract factual claims from page text (simple heuristic).
///
/// Splits into sentences, filters for informative ones (>40 chars, not questions).
fn extract_claims_from_text(text: &str, source_url: &str) -> Vec<WebClaim> {
    text.split(|c: char| c == '.' || c == '!' || c == '\n')
        .map(|s| s.trim())
        .filter(|s| s.len() > 40 && s.len() < 500)
        .filter(|s| !s.contains('?'))                // Skip questions
        .filter(|s| !s.starts_with('['))             // Skip [link] fragments
        .filter(|s| s.chars().filter(|c| c.is_alphabetic()).count() > s.len() / 2) // Mostly text
        .take(10)
        .map(|s| WebClaim {
            content: s.to_string(),
            source: source_url.to_string(),
            confidence: 0.6, // Moderate confidence for web-extracted text
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_claims_filters_short_sentences() {
        let text = "Short. This is a longer sentence that should be extracted as a factual claim from the page. Another short one.";
        let claims = extract_claims_from_text(text, "https://example.com");
        assert_eq!(claims.len(), 1);
        assert!(claims[0].content.contains("longer sentence"));
    }

    #[test]
    fn extract_claims_skips_questions() {
        let text = "What is consciousness and how does it arise in biological systems?\nConsciousness is the state of being aware of surroundings and internal mental states according to modern neuroscience research conducted across multiple laboratories";
        let claims = extract_claims_from_text(text, "https://example.com");
        assert_eq!(claims.len(), 1, "Should skip question, keep statement: {:?}", claims);
        assert!(claims[0].content.contains("aware"));
    }

    #[test]
    fn web_agent_phi_gating() {
        let safety = BrowserSafetyPolicy::default();
        let low_phi = WebAgent::new(safety.clone(), 0.1);
        assert!(!low_phi.can_navigate(), "Low Phi should not allow navigation");

        let high_phi = WebAgent::new(safety, 0.5);
        assert!(high_phi.can_navigate(), "High Phi should allow navigation");
    }

    #[test]
    fn extract_claims_limits_output() {
        // Generate many sentences
        let text: String = (0..100)
            .map(|i| format!("This is sentence number {} which is a factual claim about the world and its properties", i))
            .collect::<Vec<_>>()
            .join(". ");
        let claims = extract_claims_from_text(&text, "https://example.com");
        assert!(claims.len() <= 10, "Should cap at 10 claims, got {}", claims.len());
    }
}
