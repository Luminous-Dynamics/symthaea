// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Content Extraction Module
//!
//! Extracts clean text and claims from HTML content,
//! preparing raw web data for epistemic verification.

use anyhow::Result;
use scraper::{Html, Selector};

/// Extracted content from a web page
#[derive(Debug, Clone, Default)]
pub struct ExtractedContent {
    /// Page title
    pub title: String,
    /// Main body text (cleaned)
    pub body_text: String,
    /// Short summary (first paragraph or description)
    pub summary: String,
    /// Extracted individual claims
    pub claims: Vec<String>,
    /// Metadata (author, date, etc.)
    pub metadata: ContentMetadata,
    /// Quality score (0.0 - 1.0) based on extraction success
    pub quality: f32,
}

/// Metadata extracted from content
#[derive(Debug, Clone, Default)]
pub struct ContentMetadata {
    /// Author if found
    pub author: Option<String>,
    /// Publication date if found
    pub date: Option<String>,
    /// Site name
    pub site_name: Option<String>,
    /// Content type (article, documentation, forum, etc.)
    pub content_type: ContentType,
}

/// Types of web content
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContentType {
    /// General article
    #[default]
    Article,
    /// Technical documentation
    Documentation,
    /// Forum or Q&A
    Forum,
    /// Academic paper
    Academic,
    /// News article
    News,
    /// Blog post
    Blog,
    /// Unknown type
    Unknown,
}

/// Content extractor for web pages
#[derive(Debug, Clone)]
pub struct ContentExtractor {
    /// Minimum content length to be considered valid
    min_content_length: usize,
    /// Maximum claims to extract per page
    max_claims: usize,
}

impl Default for ContentExtractor {
    fn default() -> Self {
        Self {
            min_content_length: 50,
            max_claims: 20,
        }
    }
}

impl ContentExtractor {
    /// Create a new content extractor
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom settings
    pub fn with_config(min_content_length: usize, max_claims: usize) -> Self {
        Self {
            min_content_length,
            max_claims,
        }
    }

    /// Extract content from HTML
    pub fn extract(&self, html: &str, url: &str) -> Result<ExtractedContent> {
        let document = Html::parse_document(html);

        // Extract title
        let title = self.extract_title(&document);

        // Extract metadata
        let metadata = self.extract_metadata(&document, url);

        // Extract body text
        let body_text = self.extract_body_text(&document);

        // Extract summary
        let summary = self.extract_summary(&document, &body_text);

        // Extract individual claims
        let claims = self.extract_claims(&body_text);

        // Calculate quality score
        let quality = self.calculate_quality(&title, &body_text, &claims);

        Ok(ExtractedContent {
            title,
            body_text,
            summary,
            claims,
            metadata,
            quality,
        })
    }

    /// Extract page title
    fn extract_title(&self, document: &Html) -> String {
        // Try title tag
        if let Ok(selector) = Selector::parse("title") {
            if let Some(element) = document.select(&selector).next() {
                let title = element.text().collect::<String>().trim().to_string();
                if !title.is_empty() {
                    return title;
                }
            }
        }

        // Try h1
        if let Ok(selector) = Selector::parse("h1") {
            if let Some(element) = document.select(&selector).next() {
                let title = element.text().collect::<String>().trim().to_string();
                if !title.is_empty() {
                    return title;
                }
            }
        }

        // Try og:title
        if let Ok(selector) = Selector::parse("meta[property='og:title']") {
            if let Some(element) = document.select(&selector).next() {
                if let Some(content) = element.value().attr("content") {
                    if !content.is_empty() {
                        return content.to_string();
                    }
                }
            }
        }

        String::new()
    }

    /// Extract metadata from the page
    fn extract_metadata(&self, document: &Html, url: &str) -> ContentMetadata {
        let mut metadata = ContentMetadata::default();

        // Extract author
        for selector_str in &["meta[name='author']", "meta[property='article:author']"] {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    if let Some(content) = element.value().attr("content") {
                        metadata.author = Some(content.to_string());
                        break;
                    }
                }
            }
        }

        // Extract date
        for selector_str in &[
            "meta[property='article:published_time']",
            "meta[name='date']",
            "time[datetime]",
        ] {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    if let Some(content) = element
                        .value()
                        .attr("content")
                        .or_else(|| element.value().attr("datetime"))
                    {
                        metadata.date = Some(content.to_string());
                        break;
                    }
                }
            }
        }

        // Extract site name
        if let Ok(selector) = Selector::parse("meta[property='og:site_name']") {
            if let Some(element) = document.select(&selector).next() {
                if let Some(content) = element.value().attr("content") {
                    metadata.site_name = Some(content.to_string());
                }
            }
        }

        // Detect content type from URL and structure
        metadata.content_type = self.detect_content_type(url, document);

        metadata
    }

    /// Detect the type of content
    fn detect_content_type(&self, url: &str, document: &Html) -> ContentType {
        let url_lower = url.to_lowercase();

        // Check URL patterns
        if url_lower.contains("docs.")
            || url_lower.contains("/doc/")
            || url_lower.contains("/documentation/")
        {
            return ContentType::Documentation;
        }
        if url_lower.contains("arxiv.org") || url_lower.contains("/paper/") {
            return ContentType::Academic;
        }
        if url_lower.contains("stackoverflow.com")
            || url_lower.contains("/forum/")
            || url_lower.contains("/questions/")
        {
            return ContentType::Forum;
        }
        if url_lower.contains("blog.") || url_lower.contains("/blog/") {
            return ContentType::Blog;
        }
        if url_lower.contains("news.") || url_lower.contains("/news/") {
            return ContentType::News;
        }

        // Check document structure
        if let Ok(selector) = Selector::parse("pre code, .highlight, .code-block") {
            if document.select(&selector).count() > 2 {
                return ContentType::Documentation;
            }
        }

        ContentType::Article
    }

    /// Extract main body text
    fn extract_body_text(&self, document: &Html) -> String {
        // Try common article content selectors
        let content_selectors = [
            "article",
            "main",
            "[role='main']",
            ".content",
            ".post-content",
            ".entry-content",
            ".article-content",
            "#content",
        ];

        for selector_str in &content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let text = self.clean_text(&element.text().collect::<String>());
                    if text.len() >= self.min_content_length {
                        return text;
                    }
                }
            }
        }

        // Fall back to body, excluding nav, header, footer, script, style
        if let Ok(body_selector) = Selector::parse("body") {
            if let Some(body) = document.select(&body_selector).next() {
                // Get all paragraph text
                let mut paragraphs = Vec::new();
                if let Ok(p_selector) = Selector::parse("p") {
                    for p in body.select(&p_selector) {
                        let text = p.text().collect::<String>().trim().to_string();
                        if text.len() > 30 {
                            paragraphs.push(text);
                        }
                    }
                }
                return self.clean_text(&paragraphs.join("\n\n"));
            }
        }

        String::new()
    }

    /// Extract a summary from the content
    fn extract_summary(&self, document: &Html, body_text: &str) -> String {
        // Try meta description
        if let Ok(selector) =
            Selector::parse("meta[name='description'], meta[property='og:description']")
        {
            if let Some(element) = document.select(&selector).next() {
                if let Some(content) = element.value().attr("content") {
                    let summary = content.trim().to_string();
                    if !summary.is_empty() {
                        return summary;
                    }
                }
            }
        }

        // Fall back to first paragraph
        let paragraphs: Vec<&str> = body_text.split("\n\n").collect();
        if let Some(first_para) = paragraphs.first() {
            let summary = first_para.trim();
            if summary.len() > 50 {
                return if summary.len() > 300 {
                    format!("{}...", &summary[..297])
                } else {
                    summary.to_string()
                };
            }
        }

        String::new()
    }

    /// Extract individual claims from text
    fn extract_claims(&self, text: &str) -> Vec<String> {
        let mut claims = Vec::new();

        // Split into sentences
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        for sentence in sentences {
            let sentence = sentence.trim();

            // Skip too short or too long sentences
            if sentence.len() < 20 || sentence.len() > 300 {
                continue;
            }

            // Skip sentences that are likely not claims
            if self.is_likely_claim(sentence) {
                claims.push(sentence.to_string());

                if claims.len() >= self.max_claims {
                    break;
                }
            }
        }

        claims
    }

    /// Check if a sentence is likely a verifiable claim
    fn is_likely_claim(&self, sentence: &str) -> bool {
        let lower = sentence.to_lowercase();

        // Skip questions
        if sentence.ends_with('?') {
            return false;
        }

        // Skip imperative sentences (commands)
        let imperative_starts = ["click", "go to", "see", "note", "please", "try", "install"];
        for start in &imperative_starts {
            if lower.starts_with(start) {
                return false;
            }
        }

        // Skip common non-claims
        let non_claim_patterns = [
            "you can",
            "you should",
            "we will",
            "i think",
            "i believe",
            "in my opinion",
            "subscribe",
            "sign up",
            "copyright",
            "all rights reserved",
        ];
        for pattern in &non_claim_patterns {
            if lower.contains(pattern) {
                return false;
            }
        }

        // Prefer sentences with claim indicators
        let claim_indicators = [
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "can",
            "will",
            "because",
            "therefore",
            "shows",
            "demonstrates",
            "proves",
            "according to",
            "research",
            "study",
            "found",
            "discovered",
        ];
        for indicator in &claim_indicators {
            if lower.contains(indicator) {
                return true;
            }
        }

        // Accept if sentence has reasonable structure
        sentence.split_whitespace().count() >= 5
    }

    /// Clean and normalize extracted text
    fn clean_text(&self, text: &str) -> String {
        // Normalize whitespace
        let text = text
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        // Remove excessive newlines
        let mut result = String::new();
        let mut prev_newline = false;
        for c in text.chars() {
            if c == '\n' {
                if !prev_newline {
                    result.push('\n');
                    result.push('\n');
                }
                prev_newline = true;
            } else {
                result.push(c);
                prev_newline = false;
            }
        }

        result.trim().to_string()
    }

    /// Calculate content quality score
    fn calculate_quality(&self, title: &str, body: &str, claims: &[String]) -> f32 {
        let mut score: f32 = 0.0;

        // Title present: +0.2
        if !title.is_empty() {
            score += 0.2;
        }

        // Reasonable body length: +0.3
        if body.len() >= 500 {
            score += 0.3;
        } else if body.len() >= 200 {
            score += 0.15;
        }

        // Multiple claims extracted: +0.3
        if claims.len() >= 5 {
            score += 0.3;
        } else if claims.len() >= 2 {
            score += 0.15;
        }

        // Good text-to-code ratio (for documentation): +0.2
        let word_count = body.split_whitespace().count();
        if word_count >= 100 {
            score += 0.2;
        }

        score.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_simple_html() {
        let html = r#"
        <html>
        <head><title>Test Page</title></head>
        <body>
            <article>
                <p>Rust is a systems programming language. It provides memory safety without garbage collection. The borrow checker ensures safe memory access.</p>
            </article>
        </body>
        </html>
        "#;

        let extractor = ContentExtractor::new();
        let result = extractor.extract(html, "https://example.com/test").unwrap();

        assert_eq!(result.title, "Test Page");
        assert!(result.body_text.contains("Rust"));
        assert!(!result.claims.is_empty());
    }

    #[test]
    fn test_detect_documentation() {
        let html = "<html><body><p>Test</p></body></html>";
        let extractor = ContentExtractor::new();
        let result = extractor
            .extract(html, "https://docs.rust-lang.org/book")
            .unwrap();

        assert_eq!(result.metadata.content_type, ContentType::Documentation);
    }
}
