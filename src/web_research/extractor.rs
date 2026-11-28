// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Content Extraction Module
//!
//! Extracts clean text and claims from HTML content,
//! preparing raw web data for epistemic verification.

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use scraper::{Html, Selector};

/// One extracted claim with byte offsets into the exact `ExtractedContent.body_text` string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedClaim {
    pub text: String,
    pub byte_start: usize,
    pub byte_end: usize,
}

impl ExtractedClaim {
    /// Return the exact body-text slice represented by this claim.
    pub fn resolve<'a>(&self, body_text: &'a str) -> Option<&'a str> {
        body_text.get(self.byte_start..self.byte_end)
    }
}

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
    /// Claim text with exact byte spans into `body_text`.
    ///
    /// The offsets refer to this normalized extracted-text artifact, not the
    /// original HTML response. Hash `body_text.as_bytes()` to pin the artifact.
    pub claim_spans: Vec<ExtractedClaim>,
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

/// A candidate numeric series (table row, list of measurements, sequence
/// mentioned in prose) pulled out of extracted text.
///
/// This is a *best-effort structural* extraction — it makes no claim about
/// whether the numbers are accurate. Downstream consumers (e.g. the
/// Conjecture Engine feeder in `conjecture_feed.rs`) are responsible for
/// combining this with epistemic verification (`verifier.rs`) before
/// treating the series as trustworthy input.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericSeries {
    /// The line/sentence the numbers were found in, kept for audit trail.
    pub context: String,
    /// (x, y) pairs in the order the numbers appeared. When the context
    /// reads as explicit pairs (e.g. `(1, 2), (2, 4), (3, 6)`), `x` is the
    /// first number of each pair; otherwise `x` is the 0-based position of
    /// `y` among the numbers found on the line (an index series).
    pub values: Vec<(f64, f64)>,
    /// True if `values` came from explicit `(x, y)` pairs in the text
    /// rather than being inferred as an index series.
    pub explicit_pairs: bool,
}

/// Matches a bare numeric token, e.g. `-3`, `2.5`, `1.6e-19`.
static NUMBER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?").unwrap());

/// Matches explicit `(x, y)` pairs, e.g. `(1, 2)` or `(3.5, -7)`.
static PAIR_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"\(\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\)",
    )
    .unwrap()
});

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

        // Extract individual claims and preserve their exact byte positions
        // within the normalized body-text artifact.
        let claim_spans = self.extract_claim_spans(&body_text);
        let claims = claim_spans.iter().map(|claim| claim.text.clone()).collect();

        // Calculate quality score
        let quality = self.calculate_quality(&title, &body_text, &claims);

        Ok(ExtractedContent {
            title,
            body_text,
            summary,
            claims,
            claim_spans,
            metadata,
            quality,
        })
    }

    /// Best-effort scan for numeric sequences/tables in text: any line
    /// containing at least `min_values` numeric tokens is treated as a
    /// candidate series. Lines that contain explicit `(x, y)` pairs are
    /// interpreted as such; everything else is indexed 0..n in the order
    /// the numbers appear on the line.
    ///
    /// This is intentionally simple (line-oriented, no unit parsing) — it
    /// is meant to catch the common case of a table or list rendered as
    /// plain text after HTML extraction, not to be a general table parser.
    pub fn extract_numeric_series(&self, text: &str, min_values: usize) -> Vec<NumericSeries> {
        let mut series = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let pairs: Vec<(f64, f64)> = PAIR_RE
                .captures_iter(trimmed)
                .filter_map(|caps| {
                    let x: f64 = caps.get(1)?.as_str().parse().ok()?;
                    let y: f64 = caps.get(2)?.as_str().parse().ok()?;
                    Some((x, y))
                })
                .collect();

            if pairs.len() >= min_values {
                series.push(NumericSeries {
                    context: trimmed.to_string(),
                    values: pairs,
                    explicit_pairs: true,
                });
                continue;
            }

            let numbers: Vec<f64> = NUMBER_RE
                .find_iter(trimmed)
                .filter_map(|m| m.as_str().parse::<f64>().ok())
                .collect();

            if numbers.len() >= min_values {
                let values = numbers
                    .into_iter()
                    .enumerate()
                    .map(|(i, y)| (i as f64, y))
                    .collect();
                series.push(NumericSeries {
                    context: trimmed.to_string(),
                    values,
                    explicit_pairs: false,
                });
            }
        }
        series
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

    /// Extract claims and preserve byte spans into the exact input text.
    ///
    /// The splitter walks UTF-8 character boundaries and trims by byte length,
    /// so every emitted range is safe to resolve against `text` without
    /// searching for duplicate sentence strings after the fact.
    fn extract_claim_spans(&self, text: &str) -> Vec<ExtractedClaim> {
        let mut claims = Vec::new();
        let mut sentence_start = 0;

        for (delimiter_index, delimiter) in text.char_indices() {
            if !matches!(delimiter, '.' | '!' | '?') {
                continue;
            }

            self.push_claim_span(text, sentence_start, delimiter_index, &mut claims);
            sentence_start = delimiter_index + delimiter.len_utf8();

            if claims.len() >= self.max_claims {
                return claims;
            }
        }

        self.push_claim_span(text, sentence_start, text.len(), &mut claims);
        claims
    }

    fn push_claim_span(
        &self,
        text: &str,
        sentence_start: usize,
        sentence_end: usize,
        claims: &mut Vec<ExtractedClaim>,
    ) {
        if claims.len() >= self.max_claims || sentence_start > sentence_end {
            return;
        }
        let Some(sentence) = text.get(sentence_start..sentence_end) else {
            return;
        };
        let trimmed = sentence.trim();
        if trimmed.len() < 20 || trimmed.len() > 300 || !self.is_likely_claim(trimmed) {
            return;
        }

        let leading_trim = sentence.len() - sentence.trim_start().len();
        let byte_start = sentence_start + leading_trim;
        let byte_end = byte_start + trimmed.len();
        claims.push(ExtractedClaim {
            text: trimmed.to_owned(),
            byte_start,
            byte_end,
        });
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
    fn extracted_claim_spans_resolve_exact_trimmed_utf8_byte_ranges() {
        let extractor = ContentExtractor::new();
        let text = "Introductory context.  Rust is a systems programming language.  Café is a useful UTF-8 test case.";
        let claims = extractor.extract_claim_spans(text);

        assert_eq!(claims.len(), 2);
        for claim in &claims {
            assert_eq!(claim.resolve(text), Some(claim.text.as_str()));
            assert!(text.is_char_boundary(claim.byte_start));
            assert!(text.is_char_boundary(claim.byte_end));
            assert!(claim.byte_start < claim.byte_end);
        }
        assert_eq!(
            claims[0].resolve(text),
            Some("Rust is a systems programming language")
        );
        assert_eq!(claims[1].resolve(text), Some("Café is a useful UTF-8 test case"));
    }

    #[test]
    fn extracted_claim_spans_do_not_shift_when_duplicate_sentences_occur() {
        let extractor = ContentExtractor::new();
        let sentence = "Rust is a systems programming language";
        let text = format!("{sentence}. Neutral ornamental separator text. {sentence}!");
        let claims = extractor.extract_claim_spans(&text);

        assert_eq!(claims.len(), 2);
        assert_eq!(claims[0].text, claims[1].text);
        assert_ne!(claims[0].byte_start, claims[1].byte_start);
        assert_eq!(claims[0].resolve(&text), Some(sentence));
        assert_eq!(claims[1].resolve(&text), Some(sentence));
    }

    #[test]
    fn extract_content_claim_spans_are_bound_to_normalized_body_text() {
        let html = r#"
        <html><body><article>
            <p>Rust is a systems programming language. It provides memory safety without garbage collection.</p>
        </article></body></html>
        "#;
        let result = ContentExtractor::new()
            .extract(html, "https://example.com/research")
            .unwrap();

        assert_eq!(result.claims.len(), result.claim_spans.len());
        for (claim, span) in result.claims.iter().zip(&result.claim_spans) {
            assert_eq!(claim, &span.text);
            assert_eq!(span.resolve(&result.body_text), Some(claim.as_str()));
        }
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

    #[test]
    fn test_extract_numeric_series_index_form() {
        let extractor = ContentExtractor::new();
        let text = "The measured values were 1, 1, 2, 3, 5, 8, 13 across the trial.\nNo numbers on this line.";
        let series = extractor.extract_numeric_series(text, 3);

        assert_eq!(series.len(), 1);
        assert!(!series[0].explicit_pairs);
        assert_eq!(
            series[0].values,
            vec![
                (0.0, 1.0),
                (1.0, 1.0),
                (2.0, 2.0),
                (3.0, 3.0),
                (4.0, 5.0),
                (5.0, 8.0),
                (6.0, 13.0),
            ]
        );
    }

    #[test]
    fn test_extract_numeric_series_explicit_pairs() {
        let extractor = ContentExtractor::new();
        let text = "Observed points: (1, 2), (2, 4), (3, 6), (4, 8).";
        let series = extractor.extract_numeric_series(text, 3);

        assert_eq!(series.len(), 1);
        assert!(series[0].explicit_pairs);
        assert_eq!(
            series[0].values,
            vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)]
        );
    }

    #[test]
    fn test_extract_numeric_series_respects_min_values() {
        let extractor = ContentExtractor::new();
        let text = "Only two numbers: 1 and 2.";
        let series = extractor.extract_numeric_series(text, 3);
        assert!(series.is_empty());
    }
}
