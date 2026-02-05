//! Web Research Orchestrator
//!
//! Coordinates the research pipeline: query planning, fetching,
//! extraction, verification, and integration. This is the main
//! entry point for autonomous web research.

use anyhow::{Result, Context};
use std::time::Duration;

use super::types::{
    WebResearchResult, WebResearchConfig,
    EpistemicStatus, ResearchSource, ResearchStatus, VerifiedClaim,
};
use super::extractor::{ContentExtractor, ContentType};
use super::verifier::{EpistemicVerifier, VerificationContext, SourceEvidence};

/// Research orchestrator for autonomous web research
#[derive(Debug)]
pub struct WebResearcher {
    /// Configuration
    config: WebResearchConfig,
    /// HTTP client for fetching pages
    client: reqwest::Client,
    /// Content extractor
    extractor: ContentExtractor,
    /// Epistemic verifier
    verifier: EpistemicVerifier,
}

impl WebResearcher {
    /// Create a new web researcher with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(WebResearchConfig::default())
    }

    /// Create a new web researcher with custom configuration
    pub fn with_config(config: WebResearchConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.request_timeout_ms))
            .user_agent(&config.user_agent)
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            config,
            client,
            extractor: ContentExtractor::new(),
            verifier: EpistemicVerifier::new(),
        })
    }

    /// Research and verify a query
    ///
    /// This is the main entry point for epistemic research.
    /// Returns a research result with verified claims.
    pub async fn research_and_verify(&self, query: &str) -> Result<WebResearchResult> {
        // Build search queries
        let search_queries = self.plan_queries(query);

        // Fetch and extract from multiple sources
        let mut all_sources: Vec<SourceEvidence> = Vec::new();
        let mut all_claims: Vec<String> = Vec::new();
        let mut primary_content = String::new();
        let mut primary_title = String::new();

        for search_query in &search_queries {
            match self.fetch_search_results(search_query).await {
                Ok(urls) => {
                    for url in urls.iter().take(self.config.max_concurrent_requests) {
                        if let Ok(extraction) = self.fetch_and_extract(url).await {
                            // Build source evidence
                            let source = SourceEvidence {
                                url: url.clone(),
                                domain: self.extract_domain(url),
                                source_type: self.classify_source(url, extraction.metadata.content_type),
                                relevant_text: extraction.summary.clone(),
                                supports: Some(true), // Will be refined by verifier
                                similarity: self.calculate_similarity(&extraction.body_text, query),
                                credibility: extraction.quality,
                            };

                            all_sources.push(source);
                            all_claims.extend(extraction.claims);

                            if primary_content.is_empty() && extraction.quality > 0.5 {
                                primary_content = extraction.body_text;
                                primary_title = extraction.title;
                            }
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        // Handle no results
        if all_sources.is_empty() {
            return Ok(WebResearchResult::with_status(ResearchStatus::NoResults));
        }

        // Verify claims
        let mut verified_claims: Vec<VerifiedClaim> = Vec::new();
        for claim in all_claims.iter().take(10) {
            let context = VerificationContext {
                sources: all_sources.clone(),
                query: query.to_string(),
                domain: self.detect_domain(query),
            };
            let verification = self.verifier.verify_claim(claim, &context);
            verified_claims.push(VerifiedClaim {
                text: verification.claim,
                status: verification.status,
                confidence: verification.confidence,
                supporting_sources: verification.supporting_sources,
                contradicting_sources: verification.contradicting_sources,
                hedge: verification.hedge,
            });
        }

        // Calculate overall confidence
        let overall_confidence = if verified_claims.is_empty() {
            0.3
        } else {
            verified_claims.iter().map(|c| c.confidence).sum::<f32>() / verified_claims.len() as f32
        };

        // Determine overall epistemic status
        let epistemic_status = self.aggregate_epistemic_status(&verified_claims);

        // Generate summary
        let summary = self.generate_summary(&primary_content, query);

        Ok(WebResearchResult {
            title: primary_title,
            url: all_sources.first().map_or(String::new(), |s| s.url.clone()),
            content: primary_content,
            summary,
            source_type: all_sources.first().map_or(ResearchSource::Web, |s| s.source_type),
            relevance: self.calculate_average_relevance(&all_sources),
            confidence: overall_confidence,
            epistemic_status,
            status: ResearchStatus::Success,
            claims: verified_claims,
        })
    }

    /// Plan search queries for a user query
    fn plan_queries(&self, query: &str) -> Vec<String> {
        let mut queries = vec![query.to_string()];

        // Add domain-specific query variants
        let lower = query.to_lowercase();

        if lower.contains("rust") || lower.contains("programming") {
            queries.push(format!("{} site:rust-lang.org", query));
            queries.push(format!("{} documentation", query));
        }

        if lower.contains("nix") || lower.contains("nixos") {
            queries.push(format!("{} site:nixos.org", query));
        }

        if lower.contains("python") {
            queries.push(format!("{} site:docs.python.org", query));
        }

        queries
    }

    /// Fetch search results for a query
    ///
    /// Returns URLs to fetch. In production, this would use a search API.
    /// Currently returns mock results for demonstration.
    async fn fetch_search_results(&self, query: &str) -> Result<Vec<String>> {
        // In a real implementation, this would call a search API
        // For now, we construct likely URLs based on the query
        let mut urls = Vec::new();
        let query_lower = query.to_lowercase();

        // Add domain-specific URLs based on query content
        if query_lower.contains("rust") {
            urls.push("https://doc.rust-lang.org/book/".to_string());
            urls.push("https://rust-lang.org/".to_string());
        }

        if query_lower.contains("nix") || query_lower.contains("nixos") {
            urls.push("https://nixos.org/manual/nixos/stable/".to_string());
            urls.push("https://nixos.wiki/".to_string());
        }

        if query_lower.contains("python") {
            urls.push("https://docs.python.org/3/".to_string());
        }

        // Always include Wikipedia as a baseline
        let wiki_query = query.replace(' ', "_");
        urls.push(format!("https://en.wikipedia.org/wiki/{}", wiki_query));

        Ok(urls)
    }

    /// Fetch and extract content from a URL
    async fn fetch_and_extract(&self, url: &str) -> Result<super::extractor::ExtractedContent> {
        let response = self.client
            .get(url)
            .send()
            .await
            .context("Failed to fetch URL")?;

        if !response.status().is_success() {
            anyhow::bail!("HTTP error: {}", response.status());
        }

        let html = response
            .text()
            .await
            .context("Failed to read response body")?;

        // Limit content size
        let html = if html.len() > self.config.max_content_length {
            html[..self.config.max_content_length].to_string()
        } else {
            html
        };

        self.extractor.extract(&html, url)
    }

    /// Extract domain from URL
    fn extract_domain(&self, url: &str) -> String {
        url.split("://")
            .nth(1)
            .unwrap_or("")
            .split('/')
            .next()
            .unwrap_or("")
            .to_string()
    }

    /// Classify source type based on URL and content type
    fn classify_source(&self, url: &str, content_type: ContentType) -> ResearchSource {
        let domain = self.extract_domain(url).to_lowercase();

        if domain.contains("arxiv.org") || domain.contains("doi.org") || domain.contains("pubmed") {
            return ResearchSource::Academic;
        }

        if domain.contains("docs.") || domain.contains("doc.")
            || content_type == ContentType::Documentation {
            return ResearchSource::Documentation;
        }

        ResearchSource::Web
    }

    /// Calculate text similarity (simple word overlap)
    fn calculate_similarity(&self, text: &str, query: &str) -> f32 {
        let text_lower = text.to_lowercase();
        let text_words: std::collections::HashSet<&str> = text_lower
            .split_whitespace()
            .collect();

        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower
            .split_whitespace()
            .collect();

        if query_words.is_empty() {
            return 0.0;
        }

        let matches = query_words.iter()
            .filter(|w| text_words.contains(*w))
            .count();

        matches as f32 / query_words.len() as f32
    }

    /// Detect the domain of a query
    fn detect_domain(&self, query: &str) -> String {
        let lower = query.to_lowercase();

        if lower.contains("programming") || lower.contains("code")
            || lower.contains("rust") || lower.contains("python") {
            return "programming".to_string();
        }

        if lower.contains("nixos") || lower.contains("nix") || lower.contains("linux") {
            return "systems".to_string();
        }

        if lower.contains("science") || lower.contains("research") || lower.contains("study") {
            return "science".to_string();
        }

        "general".to_string()
    }

    /// Calculate average relevance across sources
    fn calculate_average_relevance(&self, sources: &[SourceEvidence]) -> f32 {
        if sources.is_empty() {
            return 0.0;
        }
        sources.iter().map(|s| s.similarity).sum::<f32>() / sources.len() as f32
    }

    /// Aggregate epistemic status from verified claims
    fn aggregate_epistemic_status(&self, claims: &[VerifiedClaim]) -> EpistemicStatus {
        if claims.is_empty() {
            return EpistemicStatus::InsufficientEvidence;
        }

        let high_conf = claims.iter()
            .filter(|c| c.status == EpistemicStatus::HighConfidence)
            .count();
        let contradicted = claims.iter()
            .filter(|c| c.status == EpistemicStatus::Contradicted || c.status == EpistemicStatus::False)
            .count();

        if contradicted > claims.len() / 3 {
            return EpistemicStatus::Contradicted;
        }

        if high_conf > claims.len() / 2 {
            return EpistemicStatus::HighConfidence;
        }

        let avg_confidence: f32 = claims.iter().map(|c| c.confidence).sum::<f32>() / claims.len() as f32;

        if avg_confidence > 0.7 {
            EpistemicStatus::ModerateConfidence
        } else if avg_confidence > 0.4 {
            EpistemicStatus::LowConfidence
        } else {
            EpistemicStatus::InsufficientEvidence
        }
    }

    /// Generate a summary from content
    fn generate_summary(&self, content: &str, query: &str) -> String {
        if content.is_empty() {
            return format!("No substantial content found for: {}", query);
        }

        // Take first 500 chars as summary
        let summary = if content.len() > 500 {
            format!("{}...", &content[..497])
        } else {
            content.to_string()
        };

        // Clean up
        summary.lines().take(5).collect::<Vec<_>>().join(" ")
    }

    /// Get a reference to the verifier for direct access
    pub fn verifier(&self) -> &EpistemicVerifier {
        &self.verifier
    }

    /// Get a mutable reference to the verifier
    pub fn verifier_mut(&mut self) -> &mut EpistemicVerifier {
        &mut self.verifier
    }
}

/// Error returned when WebResearcher creation fails
#[derive(Debug)]
pub struct WebResearcherCreationError(anyhow::Error);

impl std::fmt::Display for WebResearcherCreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to create WebResearcher: {}", self.0)
    }
}

impl std::error::Error for WebResearcherCreationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

impl WebResearcher {
    /// Try to create a WebResearcher with default configuration.
    ///
    /// Returns `None` if the HTTP client cannot be created (e.g., TLS initialization failure).
    /// This is a safer alternative to `Default::default()` which can panic.
    pub fn try_default() -> Option<Self> {
        Self::new().ok()
    }
}

impl Default for WebResearcher {
    /// Creates a WebResearcher with default configuration.
    ///
    /// # Panics
    ///
    /// Panics if the HTTP client cannot be created. For a non-panicking alternative,
    /// use `WebResearcher::new()` or `WebResearcher::try_default()`.
    fn default() -> Self {
        Self::new().expect("Failed to create default WebResearcher: HTTP client initialization failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_planning() {
        let researcher = WebResearcher::new().unwrap();

        let queries = researcher.plan_queries("rust memory safety");
        assert!(queries.len() >= 1);
        assert!(queries[0].contains("rust"));
    }

    #[test]
    fn test_domain_detection() {
        let researcher = WebResearcher::new().unwrap();

        assert_eq!(researcher.detect_domain("rust programming language"), "programming");
        assert_eq!(researcher.detect_domain("nixos configuration"), "systems");
        assert_eq!(researcher.detect_domain("what is the weather"), "general");
    }

    #[test]
    fn test_similarity_calculation() {
        let researcher = WebResearcher::new().unwrap();

        let sim = researcher.calculate_similarity(
            "Rust is a systems programming language focused on safety",
            "rust programming safety"
        );
        assert!(sim > 0.5);

        let sim_low = researcher.calculate_similarity(
            "The weather is nice today",
            "rust programming"
        );
        assert!(sim_low < 0.5);
    }
}
