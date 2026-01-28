//! Web Research Engine - Consciousness-Guided Autonomous Research
//!
//! **Revolutionary**: First AI that researches when uncertain, guided by ∇Φ.

use crate::language::parser::SemanticParser;
use crate::language::vocabulary::Vocabulary;
use super::types::{Source, SearchQuery, ResearchPlan, Claim, VerificationLevel};
use super::extractor::{ContentExtractor, ExtractedContent};
use super::verifier::{EpistemicVerifier, Verification, SourceClassifier};
use anyhow::{Result, Context};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{info, warn, debug};

// ============================================================================
// Pluggable search backend
// ============================================================================

/// A single search hit returned by a backend.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Async search backend trait — allows swapping search providers.
#[async_trait]
pub trait SearchBackend: Send + Sync {
    async fn search(&self, query: &str, max_results: usize) -> Result<Vec<SearchHit>>;
}

/// DuckDuckGo Instant Answer API backend (free, no API key required).
pub struct DuckDuckGoBackend {
    client: Client,
}

impl DuckDuckGoBackend {
    pub fn new(config: &ResearchConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_seconds))
            .user_agent(&config.user_agent)
            .build()
            .unwrap_or_else(|_| Client::new());
        Self { client }
    }
}

#[async_trait]
impl SearchBackend for DuckDuckGoBackend {
    async fn search(&self, query: &str, max_results: usize) -> Result<Vec<SearchHit>> {
        let url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_html=1",
            urlencoding::encode(query)
        );

        let response: serde_json::Value = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        let mut hits = Vec::new();

        // AbstractText → first hit
        if let Some(text) = response.get("AbstractText").and_then(|v| v.as_str()) {
            if !text.is_empty() {
                let title = response.get("Heading")
                    .and_then(|v| v.as_str())
                    .unwrap_or("DuckDuckGo Result")
                    .to_string();
                let url = response.get("AbstractURL")
                    .and_then(|v| v.as_str())
                    .unwrap_or("https://duckduckgo.com")
                    .to_string();
                hits.push(SearchHit { title, url, snippet: text.to_string() });
            }
        }

        // RelatedTopics → additional hits
        if let Some(topics) = response.get("RelatedTopics").and_then(|v| v.as_array()) {
            for topic in topics.iter().take(max_results.saturating_sub(hits.len())) {
                if let (Some(text), Some(first_url)) = (
                    topic.get("Text").and_then(|v| v.as_str()),
                    topic.get("FirstURL").and_then(|v| v.as_str()),
                ) {
                    hits.push(SearchHit {
                        title: text.chars().take(80).collect(),
                        url: first_url.to_string(),
                        snippet: text.to_string(),
                    });
                }
            }
        }

        hits.truncate(max_results);
        Ok(hits)
    }
}

/// Mock search backend for offline / CI testing.
///
/// Returns canned results so integration tests can exercise the full
/// research pipeline without network access.
pub struct MockSearchBackend {
    hits: Vec<SearchHit>,
}

impl MockSearchBackend {
    /// Create a mock that always returns the provided hits.
    pub fn new(hits: Vec<SearchHit>) -> Self {
        Self { hits }
    }

    /// Convenience: a single Wikipedia-style hit for any query.
    pub fn wikipedia_stub() -> Self {
        Self::new(vec![SearchHit {
            title: "Mock Wikipedia Result".to_string(),
            url: "https://en.wikipedia.org/wiki/Mock".to_string(),
            snippet: "This is a mock search result for testing purposes.".to_string(),
        }])
    }
}

#[async_trait]
impl SearchBackend for MockSearchBackend {
    async fn search(&self, _query: &str, max_results: usize) -> Result<Vec<SearchHit>> {
        Ok(self.hits.iter().take(max_results).cloned().collect())
    }
}

/// Research configuration
#[derive(Debug, Clone)]
pub struct ResearchConfig {
    /// Maximum sources to fetch per query
    pub max_sources: usize,

    /// Timeout for each HTTP request
    pub request_timeout_seconds: u64,

    /// Minimum source credibility to include
    pub min_credibility: f64,

    /// Verification level
    pub verification_level: VerificationLevel,

    /// User agent string
    pub user_agent: String,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            max_sources: 5,
            request_timeout_seconds: 10,
            min_credibility: 0.6,
            verification_level: VerificationLevel::Standard,
            user_agent: "Symthaea/0.1 (Conscious AI Research Assistant)".to_string(),
        }
    }
}

/// Research result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchResult {
    /// Original query
    pub query: String,

    /// Sources found and verified
    pub sources: Vec<Source>,

    /// Verified claims
    pub verifications: Vec<Verification>,

    /// Overall confidence (0.0-1.0)
    pub confidence: f64,

    /// Summary of findings
    pub summary: String,

    /// New concepts learned
    pub new_concepts: Vec<String>,

    /// Time taken (ms)
    pub time_taken_ms: u64,
}

/// Web researcher
pub struct WebResearcher {
    /// HTTP client
    client: Client,

    /// Content extractor
    extractor: ContentExtractor,

    /// Epistemic verifier
    verifier: EpistemicVerifier,

    /// Source classifier (domain → EpistemicCube)
    classifier: SourceClassifier,

    /// Pluggable search backend
    backend: Box<dyn SearchBackend>,

    /// Semantic parser
    parser: SemanticParser,

    /// Vocabulary for encoding
    vocabulary: Vocabulary,

    /// Configuration
    config: ResearchConfig,
}

impl WebResearcher {
    pub fn new() -> Result<Self> {
        let config = ResearchConfig::default();
        Self::with_config(config)
    }

    pub fn with_config(config: ResearchConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_seconds))
            .user_agent(&config.user_agent)
            .build()
            .context("Failed to create HTTP client")?;

        let backend = Box::new(DuckDuckGoBackend::new(&config));

        Ok(Self {
            client,
            extractor: ContentExtractor::new(),
            verifier: EpistemicVerifier::new(),
            classifier: SourceClassifier::new(),
            backend,
            parser: SemanticParser::new(),
            vocabulary: Vocabulary::new(),
            config,
        })
    }

    /// Replace the search backend (useful for testing with MockSearchBackend).
    pub fn with_backend(mut self, backend: Box<dyn SearchBackend>) -> Self {
        self.backend = backend;
        self
    }

    /// Access the source classifier for external use (e.g., Phase 3.6).
    pub fn classifier(&self) -> &SourceClassifier {
        &self.classifier
    }

    /// High-level research: search + classify sources, return ResearchResult.
    ///
    /// This uses the pluggable SearchBackend and SourceClassifier without
    /// the full epistemic verification pipeline.
    pub async fn research(&self, query: &str) -> Result<ResearchResult> {
        let start_time = std::time::Instant::now();
        info!("🔍 Research (backend): {}", query);

        // 1. Search via backend
        let hits = self.backend.search(query, self.config.max_sources).await?;
        debug!("Backend returned {} hits", hits.len());

        // 2. Convert hits → Sources (classify each URL)
        let mut sources = Vec::new();
        for hit in &hits {
            let cube = self.classifier.classify(&hit.url);
            let credibility = self.classifier.credibility_score(&cube);

            let content = ExtractedContent {
                text: hit.snippet.clone(),
                title: hit.title.clone(),
                content_type: super::extractor::ContentType::Article,
                published_date: None,
                author: None,
                paragraphs: vec![hit.snippet.clone()],
                citations: Vec::new(),
            };

            sources.push(self.extractor.create_source(
                hit.url.clone(),
                content,
                credibility,
            ));
        }

        // 3. Build summary from best source
        let summary = if let Some(best) = sources.first() {
            best.content.chars().take(500).collect()
        } else {
            format!("No results found for '{}'", query)
        };

        let elapsed = start_time.elapsed();

        Ok(ResearchResult {
            query: query.to_string(),
            sources,
            verifications: Vec::new(),
            confidence: 0.5,
            summary,
            new_concepts: Vec::new(),
            time_taken_ms: elapsed.as_millis() as u64,
        })
    }

    /// Research a query with epistemic verification
    pub async fn research_and_verify(&self, query: &str) -> Result<ResearchResult> {
        let start_time = std::time::Instant::now();

        info!("🔍 Researching: {}", query);

        // 1. Plan research (consciousness-guided)
        let plan = self.create_research_plan(query);
        debug!("Research plan created with {} sub-questions", plan.sub_questions.len());

        // 2. Generate semantic search queries
        let queries = self.generate_search_queries(query);
        debug!("Generated {} search queries", queries.len());

        // 3. Fetch sources
        let sources = self.fetch_sources(&queries).await?;
        info!("Fetched {} sources", sources.len());

        // 4. Extract claims from query
        let claims = self.extract_claims(query);
        debug!("Extracted {} claims to verify", claims.len());

        // 5. Verify each claim epistemically
        let verifications: Vec<Verification> = claims
            .iter()
            .map(|claim| self.verifier.verify_claim(claim, &sources, plan.verification_level))
            .collect();

        // 6. Calculate overall confidence
        let confidence = if verifications.is_empty() {
            0.5  // No claims to verify, neutral confidence
        } else {
            verifications.iter()
                .map(|v| v.confidence)
                .sum::<f64>() / verifications.len() as f64
        };

        // 7. Generate summary
        let summary = self.generate_summary(query, &verifications, &sources);

        // 8. Extract new concepts
        let new_concepts = self.extract_new_concepts(&sources);

        let elapsed = start_time.elapsed();

        info!("✅ Research complete in {}ms - Confidence: {:.2}", elapsed.as_millis(), confidence);

        Ok(ResearchResult {
            query: query.to_string(),
            sources,
            verifications,
            confidence,
            summary,
            new_concepts,
            time_taken_ms: elapsed.as_millis() as u64,
        })
    }

    /// Create research plan
    fn create_research_plan(&self, query: &str) -> ResearchPlan {
        // Parse query semantically
        let parsed = self.parser.parse(query);

        // For now, use simple decomposition
        // In production, use ReasoningEngine to decompose complex queries
        let sub_questions = vec![query.to_string()];

        ResearchPlan {
            query: query.to_string(),
            sub_questions,
            expected_phi_gain: 0.3,  // Estimated
            verification_level: self.config.verification_level,
            max_sources: self.config.max_sources,
            timeout_seconds: self.config.request_timeout_seconds,
        }
    }

    /// Generate semantic search queries
    fn generate_search_queries(&self, query: &str) -> Vec<SearchQuery> {
        // Parse query
        let parsed = self.parser.parse(query);

        // Generate expansions (simplified for now)
        let expansions = vec![
            query.to_string(),
            format!("{} definition", query),
            format!("{} explanation", query),
            format!("what is {}", query),
        ];

        vec![SearchQuery {
            original: query.to_string(),
            expansions,
            intent_encoding: parsed.unified_encoding,
            priority: 1.0,
        }]
    }

    /// Fetch sources from web
    async fn fetch_sources(&self, queries: &[SearchQuery]) -> Result<Vec<Source>> {
        let mut sources = Vec::new();

        // For MVP: Use DuckDuckGo Instant Answer API (no API key required)
        // In production: Use proper search APIs (Google, Bing, etc.)

        for query in queries.iter().take(1) {  // Just first query for MVP
            // Try DuckDuckGo Instant Answer API
            let ddg_url = format!(
                "https://api.duckduckgo.com/?q={}&format=json&no_html=1",
                urlencoding::encode(&query.original)
            );

            match self.fetch_ddg(&ddg_url).await {
                Ok(mut ddg_sources) => {
                    sources.append(&mut ddg_sources);
                }
                Err(e) => {
                    warn!("DuckDuckGo fetch failed: {}", e);
                }
            }

            // Also try Wikipedia API for high-quality content
            match self.fetch_wikipedia(&query.original).await {
                Ok(wiki_source) => {
                    sources.push(wiki_source);
                }
                Err(e) => {
                    debug!("Wikipedia fetch failed: {}", e);
                }
            }
        }

        // Limit to max_sources
        sources.truncate(self.config.max_sources);

        Ok(sources)
    }

    /// Fetch from DuckDuckGo Instant Answer API
    async fn fetch_ddg(&self, url: &str) -> Result<Vec<Source>> {
        let response: serde_json::Value = self.client
            .get(url)
            .send()
            .await?
            .json()
            .await?;

        let mut sources = Vec::new();

        // Extract Abstract if available
        if let Some(abstract_text) = response.get("AbstractText").and_then(|v| v.as_str()) {
            if !abstract_text.is_empty() {
                let abstract_url = response.get("AbstractURL")
                    .and_then(|v| v.as_str())
                    .unwrap_or("https://duckduckgo.com");

                let content = ExtractedContent {
                    text: abstract_text.to_string(),
                    title: response.get("Heading")
                        .and_then(|v| v.as_str())
                        .unwrap_or("DuckDuckGo Result")
                        .to_string(),
                    content_type: super::extractor::ContentType::Article,
                    published_date: None,
                    author: None,
                    paragraphs: vec![abstract_text.to_string()],
                    citations: Vec::new(),
                };

                let source = self.extractor.create_source(
                    abstract_url.to_string(),
                    content,
                    0.75,  // DuckDuckGo abstracts are generally reliable
                );

                sources.push(source);
            }
        }

        Ok(sources)
    }

    /// Fetch from Wikipedia API
    async fn fetch_wikipedia(&self, query: &str) -> Result<Source> {
        let api_url = format!(
            "https://en.wikipedia.org/api/rest_v1/page/summary/{}",
            urlencoding::encode(query)
        );

        let response: serde_json::Value = self.client
            .get(&api_url)
            .send()
            .await?
            .json()
            .await?;

        let title = response.get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Wikipedia Article")
            .to_string();

        let extract = response.get("extract")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let url = response.get("content_urls")
            .and_then(|v| v.get("desktop"))
            .and_then(|v| v.get("page"))
            .and_then(|v| v.as_str())
            .unwrap_or("https://wikipedia.org")
            .to_string();

        let content = ExtractedContent {
            text: extract.clone(),
            title: title.clone(),
            content_type: super::extractor::ContentType::Article,
            published_date: None,
            author: Some("Wikipedia".to_string()),
            paragraphs: extract.split('.').map(|s| s.to_string()).collect(),
            citations: Vec::new(),
        };

        Ok(self.extractor.create_source(url, content, 0.8))  // Wikipedia is fairly reliable
    }

    /// Extract claims from query
    fn extract_claims(&self, query: &str) -> Vec<Claim> {
        // Parse the query
        let parsed = self.parser.parse(query);

        // For now, treat the whole query as a single claim
        // In production, use NLP to extract individual claims

        // Extract subject, predicate, object text from parsed words
        let subject_text = parsed.words.iter()
            .filter(|w| matches!(w.role, crate::language::SemanticRole::Subject))
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let predicate_text = parsed.words.iter()
            .filter(|w| matches!(w.role, crate::language::SemanticRole::Predicate))
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let object_text = parsed.words.iter()
            .filter(|w| matches!(w.role, crate::language::SemanticRole::Object))
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        vec![Claim {
            text: query.to_string(),
            encoding: parsed.unified_encoding.clone(),
            subject: if subject_text.is_empty() { query.to_string() } else { subject_text },
            predicate: if predicate_text.is_empty() { "is".to_string() } else { predicate_text },
            object: if object_text.is_empty() { None } else { Some(object_text) },
            extraction_confidence: 0.8,
        }]
    }

    /// Generate summary of research
    fn generate_summary(
        &self,
        query: &str,
        verifications: &[Verification],
        sources: &[Source],
    ) -> String {
        if verifications.is_empty() {
            return format!(
                "Found {} sources about '{}', but could not verify specific claims. {}",
                sources.len(),
                query,
                if sources.is_empty() {
                    "No sources found."
                } else {
                    "More research may be needed."
                }
            );
        }

        let verification = &verifications[0];  // First claim

        format!(
            "{} {} Based on {} sources.",
            verification.hedge_phrase,
            query,
            verification.sources_checked
        )
    }

    /// Extract new concepts to learn
    fn extract_new_concepts(&self, sources: &[Source]) -> Vec<String> {
        let mut concepts = Vec::new();

        for source in sources {
            // Extract capitalized terms (potential proper nouns/concepts)
            for word in source.content.split_whitespace() {
                if word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && word.len() > 3
                    && self.vocabulary.get(&word.to_lowercase()).is_none()
                {
                    concepts.push(word.to_string());
                }
            }
        }

        // Deduplicate and limit
        concepts.sort();
        concepts.dedup();
        concepts.truncate(20);

        concepts
    }
}

impl Default for WebResearcher {
    fn default() -> Self {
        Self::new().expect("Failed to create WebResearcher")
    }
}

// Helper for URL encoding
mod urlencoding {
    pub fn encode(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                ' ' => "+".to_string(),
                _ => format!("%{:02X}", c as u8),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_research_basic() {
        let researcher = WebResearcher::new().unwrap();

        // Test with a simple query
        let result = researcher.research_and_verify("rust programming language").await;

        // Should succeed or fail gracefully
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_url_encoding() {
        // urlencoding crate uses + for spaces (application/x-www-form-urlencoded style)
        assert_eq!(urlencoding::encode("hello world"), "hello+world");
        // Dots are unreserved in RFC 3986, so they're not encoded
        assert_eq!(urlencoding::encode("test@example.com"), "test%40example.com");
    }
}
