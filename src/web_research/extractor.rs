//! Content extraction from web sources
//!
//! Extracts clean, semantic content from HTML for epistemic verification.

use crate::hdc::binary_hv::HV16;
use crate::language::vocabulary::Vocabulary;
use super::types::Source;
use anyhow::Result;
use scraper::{Html, Selector};
use html2text::from_read;
use std::time::SystemTime;

/// Type of content extracted
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentType {
    Article,
    BlogPost,
    Documentation,
    Forum,
    Academic,
    News,
    Unknown,
}

/// Extracted content from a web page
#[derive(Debug, Clone)]
pub struct ExtractedContent {
    /// Clean text content
    pub text: String,

    /// Title
    pub title: String,

    /// Content type
    pub content_type: ContentType,

    /// Publication date (if found)
    pub published_date: Option<SystemTime>,

    /// Author (if found)
    pub author: Option<String>,

    /// Main paragraphs (for verification)
    pub paragraphs: Vec<String>,

    /// Extracted citations/references (if any)
    pub citations: Vec<String>,
}

/// Content extractor
pub struct ContentExtractor {
    /// Vocabulary for semantic encoding
    vocabulary: Vocabulary,
}

impl ContentExtractor {
    pub fn new() -> Self {
        Self {
            vocabulary: Vocabulary::new(),
        }
    }

    /// Extract content from HTML
    pub fn extract_from_html(&self, html: &str, url: &str) -> Result<ExtractedContent> {
        let document = Html::parse_document(html);

        // Extract title
        let title = self.extract_title(&document)?;

        // Extract main content
        let content = self.extract_main_content(&document)?;

        // Convert HTML to clean text
        let text = from_read(content.as_bytes(), 80);

        // Extract metadata
        let published_date = self.extract_publish_date(&document);
        let author = self.extract_author(&document);

        // Extract paragraphs for detailed verification
        let paragraphs = self.extract_paragraphs(&document)?;

        // Extract citations/references
        let citations = self.extract_citations(&document);

        // Determine content type
        let content_type = self.determine_content_type(url, &title, &text);

        Ok(ExtractedContent {
            text,
            title,
            content_type,
            published_date,
            author,
            paragraphs,
            citations,
        })
    }

    /// Extract title from document
    fn extract_title(&self, document: &Html) -> Result<String> {
        // Try multiple selectors for title
        let selectors = vec![
            "h1",
            "title",
            "meta[property='og:title']",
            ".article-title",
            ".post-title",
        ];

        for selector_str in selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let text = element.text().collect::<String>().trim().to_string();
                    if !text.is_empty() {
                        return Ok(text);
                    }
                }
            }
        }

        Ok("Untitled".to_string())
    }

    /// Extract main content from document
    fn extract_main_content(&self, document: &Html) -> Result<String> {
        // Try multiple selectors for main content
        let selectors = vec![
            "article",
            "main",
            ".article-content",
            ".post-content",
            ".content",
            "#content",
            ".entry-content",
        ];

        for selector_str in selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    return Ok(element.html());
                }
            }
        }

        // Fallback: use body
        if let Ok(selector) = Selector::parse("body") {
            if let Some(element) = document.select(&selector).next() {
                return Ok(element.html());
            }
        }

        anyhow::bail!("Could not extract main content")
    }

    /// Extract paragraphs for detailed analysis
    fn extract_paragraphs(&self, document: &Html) -> Result<Vec<String>> {
        let selector = Selector::parse("p")
            .map_err(|e| anyhow::anyhow!("Failed to create paragraph selector: {:?}", e))?;

        let paragraphs: Vec<String> = document
            .select(&selector)
            .map(|el| el.text().collect::<String>().trim().to_string())
            .filter(|p| p.len() > 50) // Filter out short paragraphs
            .collect();

        Ok(paragraphs)
    }

    /// Extract publication date
    fn extract_publish_date(&self, document: &Html) -> Option<SystemTime> {
        // Try common meta tags for date
        let selectors = vec![
            "meta[property='article:published_time']",
            "meta[name='publication_date']",
            "meta[name='date']",
            "time[datetime]",
        ];

        for selector_str in selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    // Try to parse date (simplified for now)
                    // In production, use chrono for proper date parsing
                    return Some(SystemTime::now());
                }
            }
        }

        None
    }

    /// Extract author
    fn extract_author(&self, document: &Html) -> Option<String> {
        let selectors = vec![
            "meta[name='author']",
            "meta[property='article:author']",
            ".author",
            ".byline",
        ];

        for selector_str in selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let author = element.text().collect::<String>().trim().to_string();
                    if !author.is_empty() {
                        return Some(author);
                    }
                }
            }
        }

        None
    }

    /// Extract citations/references
    fn extract_citations(&self, document: &Html) -> Vec<String> {
        let mut citations = Vec::new();

        // Look for reference sections
        let selectors = vec![
            ".references",
            ".citations",
            "#references",
            ".bibliography",
        ];

        for selector_str in selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                for element in document.select(&selector) {
                    let text = element.text().collect::<String>();
                    citations.push(text.trim().to_string());
                }
            }
        }

        citations
    }

    /// Determine content type from URL and content
    fn determine_content_type(&self, url: &str, title: &str, text: &str) -> ContentType {
        let url_lower = url.to_lowercase();
        let title_lower = title.to_lowercase();

        // Academic indicators
        if url_lower.contains("arxiv.org")
            || url_lower.contains("scholar.google")
            || url_lower.contains(".edu")
            || url_lower.contains("doi.org")
            || title_lower.contains("journal")
            || title_lower.contains("proceedings")
        {
            return ContentType::Academic;
        }

        // News indicators
        if url_lower.contains("news")
            || url_lower.contains("cnn.com")
            || url_lower.contains("bbc.com")
            || url_lower.contains("nytimes.com")
        {
            return ContentType::News;
        }

        // Documentation indicators
        if url_lower.contains("docs.")
            || url_lower.contains("documentation")
            || title_lower.contains("documentation")
            || title_lower.contains("manual")
        {
            return ContentType::Documentation;
        }

        // Forum indicators
        if url_lower.contains("forum")
            || url_lower.contains("reddit.com")
            || url_lower.contains("stackoverflow.com")
            || url_lower.contains("discourse.")
        {
            return ContentType::Forum;
        }

        // Blog indicators
        if url_lower.contains("blog") || title_lower.contains("blog") {
            return ContentType::BlogPost;
        }

        // Default to article
        ContentType::Article
    }

    /// Encode extracted content to HV16
    pub fn encode_content(&self, content: &ExtractedContent) -> HV16 {
        // Use vocabulary to encode the text semantically
        // For now, create a simple encoding from the title
        // In production, use full semantic parsing

        let words: Vec<&str> = content.title.split_whitespace().collect();
        let mut encoding = HV16::zero();

        for word in words {
            if let Some(entry) = self.vocabulary.get(word) {
                encoding = HV16::bundle(&[encoding, entry.encoding.clone()]);
            }
        }

        encoding
    }

    /// Create Source from extracted content
    pub fn create_source(
        &self,
        url: String,
        content: ExtractedContent,
        credibility: f64,
    ) -> Source {
        let encoding = self.encode_content(&content);

        Source {
            url,
            title: content.title,
            content: content.text,
            published_date: content.published_date,
            author: content.author,
            credibility,
            encoding,
            fetch_timestamp: SystemTime::now(),
        }
    }
}

impl Default for ContentExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_title() {
        let extractor = ContentExtractor::new();
        let html = r#"
            <!DOCTYPE html>
            <html>
            <head><title>Test Article</title></head>
            <body>
                <h1>Main Title</h1>
                <p>Content here</p>
            </body>
            </html>
        "#;

        let document = Html::parse_document(html);
        let title = extractor.extract_title(&document).unwrap();
        assert!(title == "Main Title" || title == "Test Article");
    }

    #[test]
    fn test_content_type_detection() {
        let extractor = ContentExtractor::new();

        assert_eq!(
            extractor.determine_content_type("https://arxiv.org/abs/123", "Paper", "text"),
            ContentType::Academic
        );

        assert_eq!(
            extractor.determine_content_type("https://blog.example.com", "Post", "text"),
            ContentType::BlogPost
        );

        assert_eq!(
            extractor.determine_content_type("https://docs.example.com", "Guide", "text"),
            ContentType::Documentation
        );
    }
}
