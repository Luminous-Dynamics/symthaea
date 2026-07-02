// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HTTP networking for Prism.
//!
//! Fetches web pages via `reqwest`, handles decompression, and parses them
//! into `DomTree` via `prism-dom`.

use prism_dom::DomTree;
use reqwest::header::HeaderMap;
use thiserror::Error;
use url::Url;

const PRISM_USER_AGENT: &str = "Prism/0.1 (pure Rust; +https://luminousdynamics.org)";

#[derive(Error, Debug)]
pub enum FetchError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Invalid URL: {0}")]
    InvalidUrl(#[from] url::ParseError),
    #[error("Response too large: {size} bytes (max {max})")]
    TooLarge { size: u64, max: u64 },
}

/// Metadata about a fetched page, used by the reflex arc and privacy classifier.
#[derive(Debug, Clone)]
pub struct FetchMetadata {
    /// Final URL after redirects.
    pub url: Url,
    /// HTTP status code.
    pub status: u16,
    /// Response headers.
    pub headers: HeaderMap,
    /// Content-Type header value.
    pub content_type: Option<String>,
    /// Whether the response had authentication-related headers.
    pub has_auth_headers: bool,
    /// Whether cookies were present.
    pub has_cookies: bool,
}

/// Result of fetching and parsing a page.
#[derive(Debug)]
pub struct FetchedPage {
    /// Parsed DOM tree.
    pub dom: DomTree,
    /// Raw HTML body.
    pub html: String,
    /// Fetch metadata for reflex arc / privacy classification.
    pub metadata: FetchMetadata,
}

/// HTTP client for Prism (networking layer).
pub struct PrismClient {
    client: reqwest::Client,
    max_body_size: u64,
}

impl PrismClient {
    /// Create a new client with default settings.
    pub fn new() -> Self {
        Self::with_max_body_size(10 * 1024 * 1024) // 10MB max
    }

    /// Create a new client with a specified maximum body size.
    pub fn with_max_body_size(max_body_size: u64) -> Self {
        let client = reqwest::Client::builder()
            .user_agent(PRISM_USER_AGENT)
            .gzip(true)
            .brotli(true)
            .redirect(reqwest::redirect::Policy::limited(10))
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client");

        Self {
            client,
            max_body_size,
        }
    }

    /// Fetch a URL and parse it into a DOM tree.
    pub async fn fetch(&self, url: &str) -> Result<FetchedPage, FetchError> {
        let parsed_url = Url::parse(url)?;
        log::info!("Fetching: {}", parsed_url);

        let response = self.client.get(parsed_url.clone()).send().await?;

        let status = response.status().as_u16();
        let headers = response.headers().clone();

        // Check for auth/cookie signals (used by privacy classifier)
        let has_auth_headers = headers.contains_key("www-authenticate")
            || headers.contains_key("authorization")
            || headers.contains_key("x-csrf-token");
        let has_cookies = headers.contains_key("set-cookie");

        let content_type = headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        // Check content length before downloading
        if let Some(len) = response.content_length()
            && len > self.max_body_size
        {
            return Err(FetchError::TooLarge {
                size: len,
                max: self.max_body_size,
            });
        }

        // Get final URL after redirects
        let final_url = response.url().clone();

        let html = response.text().await?;

        // Parse HTML into DOM
        let dom = prism_dom::parse_html(&html);

        let metadata = FetchMetadata {
            url: final_url,
            status,
            headers,
            content_type,
            has_auth_headers,
            has_cookies,
        };

        Ok(FetchedPage {
            dom,
            html,
            metadata,
        })
    }
}

impl Default for PrismClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_creates_successfully() {
        let _client = PrismClient::new();
    }

    #[tokio::test]
    async fn fetch_example_dot_com() {
        let client = PrismClient::new();
        // This test requires network access
        match client.fetch("https://example.com").await {
            Ok(page) => {
                assert_eq!(page.metadata.status, 200);
                assert!(!page.dom.is_empty());
                let text = page.dom.extract_visible_text();
                assert!(text.contains("Example Domain"));
                let title = page.dom.title();
                assert_eq!(title, Some("Example Domain".to_string()));
            }
            Err(e) => {
                // Network tests may fail in CI; that's OK
                eprintln!("Network test skipped: {}", e);
            }
        }
    }
}
