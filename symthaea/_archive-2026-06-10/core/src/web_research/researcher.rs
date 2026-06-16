// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Web Research Orchestrator
//!
//! Coordinates the research pipeline: query planning, fetching,
//! extraction, verification, and integration. This is the main
//! entry point for autonomous web research.

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashSet;
use std::time::Duration;

use super::extractor::{ContentExtractor, ContentType};
use super::types::{
    EpistemicStatus, ResearchSource, ResearchStatus, VerifiedClaim, WebResearchConfig,
    WebResearchResult,
};
use super::verifier::{EpistemicVerifier, SourceEvidence, VerificationContext};

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

        let mut visited_urls = std::collections::HashSet::new();

        for search_query in &search_queries {
            match self.fetch_search_results(search_query).await {
                Ok(urls) => {
                    for url in urls.iter().take(self.config.max_concurrent_requests) {
                        // Skip if this URL has already been processed during this loop cycle
                        if !visited_urls.insert(url.clone()) {
                            continue;
                        }
                        println!(
                            "\x1b[34m    [🛰️ CRAWL] Fetching semantic layer from: {}\x1b[0m",
                            url
                        );
                        if let Ok(extraction) = self.fetch_and_extract(url).await {
                            println!(
                                "\x1b[32m    [🧠 EXTRACT] Harvested {} raw claims (Structural Quality: {:.2})\x1b[0m",
                                extraction.claims.len(),
                                extraction.quality
                            );
                            // Build source evidence
                            let source = SourceEvidence {
                                url: url.clone(),
                                domain: self.extract_domain(url),
                                source_type: self
                                    .classify_source(url, extraction.metadata.content_type),
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
            // Hardened script/boilerplate filter matrix
            let claim_lower = claim.to_lowercase();
            if claim_lower.contains("@type")
                || claim_lower.contains("breadcrumb")
                || claim_lower.contains("itemlist")
                || claim_lower.contains(".push([")
                || claim_lower.contains("window.__")
                || claim_lower.contains("select citation style")
                || claim_lower.contains("table of contents")
            {
                continue;
            }
            let context = VerificationContext {
                sources: all_sources.clone(),
                query: query.to_string(),
                domain: self.detect_domain(query),
            };
            let verification = self.verifier.verify_claim(claim, &context);
            println!(
                "    \x1b[35m[⚖️ VERIFY] Status: {:?} (Confidence: {:.2}) -> \"{}\"\x1b[0m",
                verification.status, verification.confidence, claim
            );
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
            source_type: all_sources
                .first()
                .map_or(ResearchSource::Web, |s| s.source_type),
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

        if lower.contains("state space")
            || lower.contains("ssm")
            || lower.contains("liquid time")
            || lower.contains("mamba")
            || lower.contains("selective scan")
        {
            queries.push(format!("{} arxiv", query));
            queries.push(format!("{} research paper", query));
            queries.push(format!("{} openreview", query));
        }

        // Relational Manifold Intersection Vectoring
        if lower.contains("luminous dynamics")
            || lower.contains("tristan stoltz")
            || lower.contains("evolving resonant")
            || lower.contains("mycelix")
        {
            queries.push("Luminous Dynamics Tristan Stoltz".to_string());
            queries.push("luminous-nix natural language interface NixOS".to_string());
            queries.push("Evolving Resonant Co-creationism framework".to_string());
        }

        queries
    }

    /// Fetch search results for a query
    ///
    /// Returns URLs to fetch. In production, this would use a search API.
    /// Improved heuristic targeting high-quality technical domains.
    async fn fetch_search_results(&self, query: &str) -> Result<Vec<String>> {
        let mut urls = Vec::new();
        let query_lower = query.to_lowercase();
        let encoded_query = urlencoding::encode(query);

        // 0. External search provider (optional)
        if let Ok(provider_urls) = self.search_with_provider(query).await {
            urls.extend(provider_urls);
        }

        if self.config.enable_arxiv_api
            && (query_lower.contains("arxiv")
                || query_lower.contains("paper")
                || query_lower.contains("research")
                || query_lower.contains("state space")
                || query_lower.contains("ssm")
                || query_lower.contains("liquid time")
                || query_lower.contains("mamba")
                || query_lower.contains("selective scan"))
        {
            if let Ok(arxiv_urls) = self.search_arxiv_api(query).await {
                urls.extend(arxiv_urls);
            }
        }

        if self.config.enable_openreview_api
            && (query_lower.contains("openreview")
                || query_lower.contains("paper")
                || query_lower.contains("research")
                || query_lower.contains("state space")
                || query_lower.contains("ssm")
                || query_lower.contains("liquid time")
                || query_lower.contains("mamba")
                || query_lower.contains("selective scan"))
        {
            if let Ok(or_urls) = self.search_openreview_api(query).await {
                urls.extend(or_urls);
            }
        }

        // 1. Target high-authority technical domains
        if query_lower.contains("rust") {
            urls.push("https://doc.rust-lang.org/book/".to_string());
            urls.push("https://docs.rs/".to_string());
            urls.push(format!(
                "https://github.com/search?q={}+language%3ARust",
                encoded_query
            ));
        }

        if query_lower.contains("nix") || query_lower.contains("nixos") {
            urls.push("https://nixos.org/manual/nixos/stable/".to_string());
            urls.push("https://nixos.wiki/".to_string());
            urls.push("https://search.nixos.org/packages".to_string());
        }

        if query_lower.contains("hdc") || query_lower.contains("hypervector") {
            urls.push("https://en.wikipedia.org/wiki/Hyperdimensional_computing".to_string());
            urls.push("https://arxiv.org/abs/2112.15425".to_string()); // Standard HDC reference
        }

        if query_lower.contains("python") {
            urls.push("https://docs.python.org/3/".to_string());
            urls.push("https://pypi.org/".to_string());
        }

        if query_lower.contains("science")
            || query_lower.contains("research")
            || query_lower.contains("paper")
        {
            urls.push("https://arxiv.org/search/".to_string());
            urls.push("https://scholar.google.com/".to_string());
        }

        if query_lower.contains("kernel") || query_lower.contains("linux") {
            urls.push("https://www.kernel.org/doc/html/latest/".to_string());
        }

        if query_lower.contains("state space")
            || query_lower.contains("ssm")
            || query_lower.contains("liquid time")
            || query_lower.contains("mamba")
            || query_lower.contains("selective scan")
        {
            urls.push(format!(
                "https://arxiv.org/search/?query={}&searchtype=all&source=header",
                encoded_query
            ));
            urls.push(format!(
                "https://openreview.net/search?term={}",
                encoded_query
            ));
        }

        // 2. Fetch DuckDuckGo HTML results page and extract the true organic links
        let ddg_url = format!("https://html.duckduckgo.com/html/?q={}", encoded_query);
        println!("\x1b[34m    [🛰️ SEARCH] Querying fallback engine for organic links...\x1b[0m");

        if let Ok(response) = self.client.get(&ddg_url).send().await {
            if let Ok(html_text) = response.text().await {
                let doc = scraper::Html::parse_document(&html_text);
                if let Ok(selector) = scraper::Selector::parse("a.result__url") {
                    for element in doc.select(&selector) {
                        if let Some(href) = element.value().attr("href") {
                            // Verify it is an absolute external target link
                            if !href.contains("duckduckgo.com") && href.starts_with("http") {
                                urls.push(href.to_string());
                            }
                        }
                    }
                }
            }
        }

        // 3. Always include Wikipedia and StackOverflow search
        let wiki_query = query.replace(' ', "_");
        urls.push(format!("https://en.wikipedia.org/wiki/{}", wiki_query));
        urls.push(format!(
            "https://stackoverflow.com/search?q={}",
            encoded_query
        ));

        let mut seen = HashSet::new();
        urls.retain(|url| seen.insert(url.clone()));

        Ok(urls)
    }

    async fn search_with_provider(&self, query: &str) -> Result<Vec<String>> {
        let provider = match self.config.search_provider.as_deref() {
            Some(provider) => provider.trim().to_lowercase(),
            None => return Ok(Vec::new()),
        };

        match provider.as_str() {
            "serper" => self.search_serper(query).await,
            "brave" => self.search_brave(query).await,
            "searxng" | "searx" => self.search_searxng(query).await,
            _ => Ok(Vec::new()),
        }
    }

    async fn search_serper(&self, query: &str) -> Result<Vec<String>> {
        let api_key = match self.config.search_api_key.as_deref() {
            Some(key) if !key.is_empty() => key,
            _ => return Ok(Vec::new()),
        };
        let endpoint = self
            .config
            .search_endpoint
            .as_deref()
            .unwrap_or("https://google.serper.dev/search");

        let body = serde_json::json!({
            "q": query,
            "num": 10
        });

        let response = self
            .client
            .post(endpoint)
            .header("X-API-KEY", api_key)
            .header("Accept", "application/json")
            .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .json(&body)
            .send()
            .await
            .context("Serper search request failed")?;

        let json: Value = response.json().await.context("Serper JSON parse failed")?;
        Ok(extract_links(&json, &["organic"], "link"))
    }

    async fn search_brave(&self, query: &str) -> Result<Vec<String>> {
        let api_key = match self.config.search_api_key.as_deref() {
            Some(key) if !key.is_empty() => key,
            _ => return Ok(Vec::new()),
        };
        let endpoint = self
            .config
            .search_endpoint
            .as_deref()
            .unwrap_or("https://api.search.brave.com/res/v1/web/search");
        let url = format!("{endpoint}?q={}", urlencoding::encode(query));

        let response = self
            .client
            .get(url)
            .header("X-Subscription-Token", api_key)
            .header("Accept", "application/json")
            .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .send()
            .await
            .context("Brave search request failed")?;

        let json: Value = response.json().await.context("Brave JSON parse failed")?;
        Ok(extract_links(&json, &["web", "results"], "url"))
    }

    async fn search_searxng(&self, query: &str) -> Result<Vec<String>> {
        let endpoint = self
            .config
            .search_endpoint
            .as_deref()
            .unwrap_or("https://searx.be");
        let endpoint = endpoint.trim_end_matches('/');
        let url = format!(
            "{}/search?q={}&format=json",
            endpoint,
            urlencoding::encode(query)
        );

        let response = self
            .client
            .get(url)
            .header("Accept", "application/json")
            .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .send()
            .await
            .context("SearXNG search request failed")?;

        let json: Value = response.json().await.context("SearXNG JSON parse failed")?;
        Ok(extract_links(&json, &["results"], "url"))
    }

    async fn search_arxiv_api(&self, query: &str) -> Result<Vec<String>> {
        // Distill query to remove token pollution from academic indexing
        let cleaned_query = query
            .to_lowercase()
            .replace("arxiv", "")
            .replace("papers", "")
            .replace("paper", "")
            .replace("search", "")
            .replace("latest", "")
            .replace("status", "")
            .replace("architecture", "")
            .trim()
            .to_string();

        // Target both titles and abstracts specifically for high-density accuracy
        let search_query = format!("ti:\"{0}\" OR abs:\"{0}\"", cleaned_query);
        let encoded_query = urlencoding::encode(&search_query);

        // Force chronological sorting to capture state-of-the-art results (2024-2026)
        let url = format!(
            "{}?search_query={}&id_list=&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending",
            self.config.arxiv_endpoint, encoded_query
        );

        let response = self
            .client
            .get(url)
            .send()
            .await
            .context("arXiv API request failed")?;

        let body = response.text().await.context("arXiv API read failed")?;
        Ok(extract_arxiv_ids(&body))
    }

    async fn search_openreview_api(&self, query: &str) -> Result<Vec<String>> {
        let endpoint = self.config.openreview_endpoint.trim_end_matches('/');
        let encoded_query = urlencoding::encode(query);
        let url = format!("{endpoint}?term={encoded_query}");

        let mut request = self.client.get(url).header("Accept", "application/json")
            .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36");
        if let Some(token) = self.config.openreview_token.as_deref() {
            request = request.header("Cookie", format!("openreview.accessToken={token}"));
        }

        let response = request.send().await.context("OpenReview search failed")?;
        let json: Value = response
            .json()
            .await
            .context("OpenReview JSON parse failed")?;

        Ok(extract_openreview_ids(&json))
    }

    /// Fetch and extract content from a URL
    async fn fetch_and_extract(&self, url: &str) -> Result<super::extractor::ExtractedContent> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .context("Failed to fetch URL")?;

        if !response.status().is_success() {
            anyhow::bail!("HTTP error: {}", response.status());
        }

        // Prevent ingestion of binary document frameworks (PDFs, archives, applications)
        if let Some(content_type) = response.headers().get("content-type") {
            if let Ok(ct_str) = content_type.to_str() {
                let ct_lower = ct_str.to_lowercase();
                if ct_lower.contains("pdf")
                    || ct_lower.contains("octet-stream")
                    || ct_lower.contains("zip")
                {
                    anyhow::bail!("Aborting ingestion of binary non-text asset: {}", ct_str);
                }
            }
        }

        let html = response
            .text()
            .await
            .context("Failed to read response body")?;

        // Limit content size safely using character boundaries (Rust 1.76+)
        let html = if html.len() > self.config.max_content_length {
            let safe_limit = html.floor_char_boundary(self.config.max_content_length);
            html[..safe_limit].to_string()
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

        if domain.contains("docs.")
            || domain.contains("doc.")
            || content_type == ContentType::Documentation
        {
            return ResearchSource::Documentation;
        }

        ResearchSource::Web
    }

    /// Calculate text similarity (simple word overlap)
    fn calculate_similarity(&self, text: &str, query: &str) -> f32 {
        let text_lower = text.to_lowercase();
        let text_words: std::collections::HashSet<&str> = text_lower.split_whitespace().collect();

        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        if query_words.is_empty() {
            return 0.0;
        }

        let matches = query_words
            .iter()
            .filter(|w| text_words.contains(*w))
            .count();

        matches as f32 / query_words.len() as f32
    }

    /// Detect the domain of a query
    fn detect_domain(&self, query: &str) -> String {
        let lower = query.to_lowercase();

        if lower.contains("programming")
            || lower.contains("code")
            || lower.contains("rust")
            || lower.contains("python")
        {
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

        let high_conf = claims
            .iter()
            .filter(|c| c.status == EpistemicStatus::HighConfidence)
            .count();
        let contradicted = claims
            .iter()
            .filter(|c| {
                c.status == EpistemicStatus::Contradicted || c.status == EpistemicStatus::False
            })
            .count();

        if contradicted > claims.len() / 3 {
            return EpistemicStatus::Contradicted;
        }

        if high_conf > claims.len() / 2 {
            return EpistemicStatus::HighConfidence;
        }

        let avg_confidence: f32 =
            claims.iter().map(|c| c.confidence).sum::<f32>() / claims.len() as f32;

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

fn extract_links(json: &Value, path: &[&str], key: &str) -> Vec<String> {
    let mut cursor = json;
    for segment in path {
        if let Some(next) = cursor.get(*segment) {
            cursor = next;
        } else {
            return Vec::new();
        }
    }

    let mut urls = Vec::new();
    if let Some(entries) = cursor.as_array() {
        for entry in entries {
            if let Some(url) = entry.get(key).and_then(|v| v.as_str()) {
                urls.push(url.to_string());
            }
        }
    }

    urls
}

fn extract_arxiv_ids(feed: &str) -> Vec<String> {
    let mut urls = Vec::new();
    for chunk in feed.split("<entry>").skip(1) {
        if let Some(start) = chunk.find("<id>") {
            if let Some(end) = chunk[start + 4..].find("</id>") {
                let raw = &chunk[start + 4..start + 4 + end];
                let trimmed = raw.trim();
                if trimmed.contains("arxiv.org/abs/") {
                    let url = trimmed.replace("http://", "https://");
                    urls.push(url);
                }
            }
        }
    }
    urls
}

fn extract_openreview_ids(json: &Value) -> Vec<String> {
    let mut urls = Vec::new();

    let candidates = if let Some(notes) = json.get("notes").and_then(|v| v.as_array()) {
        notes
    } else if let Some(data) = json.get("data").and_then(|v| v.as_array()) {
        data
    } else if let Some(arr) = json.as_array() {
        arr
    } else {
        return urls;
    };

    for entry in candidates {
        let id = entry
            .get("id")
            .and_then(|v| v.as_str())
            .or_else(|| {
                entry
                    .get("note")
                    .and_then(|v| v.get("id"))
                    .and_then(|v| v.as_str())
            })
            .or_else(|| entry.get("forum").and_then(|v| v.as_str()));
        if let Some(id) = id {
            urls.push(format!("https://openreview.net/forum?id={id}"));
        }
    }

    urls
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
        Self::new()
            .expect("Failed to create default WebResearcher: HTTP client initialization failed")
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

        assert_eq!(
            researcher.detect_domain("rust programming language"),
            "programming"
        );
        assert_eq!(researcher.detect_domain("nixos configuration"), "systems");
        assert_eq!(researcher.detect_domain("what is the weather"), "general");
    }

    #[test]
    fn test_similarity_calculation() {
        let researcher = WebResearcher::new().unwrap();

        let sim = researcher.calculate_similarity(
            "Rust is a systems programming language focused on safety",
            "rust programming safety",
        );
        assert!(sim > 0.5);

        let sim_low =
            researcher.calculate_similarity("The weather is nice today", "rust programming");
        assert!(sim_low < 0.5);
    }
}
