// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! External search engine API clients for the comparison view.
//!
//! - Wikipedia: direct WASM fetch (CORS allowed)
//! - DuckDuckGo: via CORS proxy at :8131
//! - Ollama: direct to user's localhost:11434 (graceful fallback)

use serde::Deserialize;

/// Status of an external search query.
#[derive(Clone, Debug, PartialEq)]
pub enum SearchStatus {
    Idle,
    Loading,
    Done,
    Error(String),
}

/// A single result from an external search engine.
#[derive(Clone, Debug)]
pub struct ExternalHit {
    pub title: String,
    pub snippet: String,
    pub url: Option<String>,
    pub source_type: String, // "link", "article", "ai-generated", "instant-answer"
}

/// Results from one external search engine.
#[derive(Clone, Debug)]
pub struct ExternalResult {
    pub engine: String,
    pub hits: Vec<ExternalHit>,
    pub status: SearchStatus,
}

impl ExternalResult {
    pub fn loading(engine: &str) -> Self {
        Self {
            engine: engine.to_string(),
            hits: vec![],
            status: SearchStatus::Loading,
        }
    }
    pub fn error(engine: &str, msg: &str) -> Self {
        Self {
            engine: engine.to_string(),
            hits: vec![],
            status: SearchStatus::Error(msg.to_string()),
        }
    }
    pub fn done(engine: &str, hits: Vec<ExternalHit>) -> Self {
        Self {
            engine: engine.to_string(),
            hits,
            status: SearchStatus::Done,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// WIKIPEDIA — Direct WASM fetch (CORS allowed with origin=*)
// ═══════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct WikiResponse {
    pages: Option<Vec<WikiPage>>,
}

#[derive(Deserialize)]
struct WikiPage {
    title: Option<String>,
    excerpt: Option<String>,
    key: Option<String>,
}

pub async fn search_wikipedia(query: &str) -> ExternalResult {
    let url = format!(
        "https://en.wikipedia.org/w/rest.php/v1/search/page?q={}&limit=5&origin=*",
        urlencoding(query)
    );

    match gloo_net::http::Request::get(&url).send().await {
        Ok(resp) => match resp.json::<WikiResponse>().await {
            Ok(data) => {
                let hits: Vec<ExternalHit> = data
                    .pages
                    .unwrap_or_default()
                    .into_iter()
                    .map(|p| {
                        let title = p.title.unwrap_or_default();
                        let snippet = strip_html_tags(&p.excerpt.unwrap_or_default());
                        let key = p.key.unwrap_or_default();
                        ExternalHit {
                            title: title.clone(),
                            snippet,
                            url: Some(format!("https://en.wikipedia.org/wiki/{}", key)),
                            source_type: "article".to_string(),
                        }
                    })
                    .collect();
                ExternalResult::done("Wikipedia", hits)
            }
            Err(e) => ExternalResult::error("Wikipedia", &format!("Parse error: {}", e)),
        },
        Err(e) => ExternalResult::error("Wikipedia", &format!("Fetch failed: {}", e)),
    }
}

// ═══════════════════════════════════════════════════════════════
// DUCKDUCKGO — Via CORS proxy at :8131/api/ddg
// ═══════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct DdgResponse {
    #[serde(rename = "AbstractText")]
    abstract_text: Option<String>,
    #[serde(rename = "AbstractSource")]
    abstract_source: Option<String>,
    #[serde(rename = "AbstractURL")]
    abstract_url: Option<String>,
    #[serde(rename = "Heading")]
    heading: Option<String>,
    #[serde(rename = "Answer")]
    answer: Option<String>,
    #[serde(rename = "Definition")]
    definition: Option<String>,
    #[serde(rename = "RelatedTopics")]
    related_topics: Option<Vec<DdgTopic>>,
}

#[derive(Deserialize)]
struct DdgTopic {
    #[serde(rename = "Text")]
    text: Option<String>,
    #[serde(rename = "FirstURL")]
    first_url: Option<String>,
}

pub async fn search_duckduckgo(query: &str) -> ExternalResult {
    // Route through CORS proxy
    let proxy_base = get_proxy_base();
    let url = format!("{}/api/ddg?q={}", proxy_base, urlencoding(query));

    match gloo_net::http::Request::get(&url).send().await {
        Ok(resp) => match resp.json::<DdgResponse>().await {
            Ok(data) => {
                let mut hits = Vec::new();

                // Main abstract/answer
                if let Some(text) = data.abstract_text.filter(|t| !t.is_empty()) {
                    hits.push(ExternalHit {
                        title: data.heading.unwrap_or_else(|| "Instant Answer".to_string()),
                        snippet: text,
                        url: data.abstract_url,
                        source_type: "instant-answer".to_string(),
                    });
                } else if let Some(answer) = data.answer.filter(|a| !a.is_empty()) {
                    hits.push(ExternalHit {
                        title: "Answer".to_string(),
                        snippet: answer,
                        url: None,
                        source_type: "instant-answer".to_string(),
                    });
                } else if let Some(def) = data.definition.filter(|d| !d.is_empty()) {
                    hits.push(ExternalHit {
                        title: "Definition".to_string(),
                        snippet: def,
                        url: None,
                        source_type: "instant-answer".to_string(),
                    });
                }

                // Related topics (up to 3)
                if let Some(topics) = data.related_topics {
                    for topic in topics.into_iter().take(3) {
                        if let Some(text) = topic.text.filter(|t| !t.is_empty()) {
                            hits.push(ExternalHit {
                                title: text[..text.len().min(60)].to_string(),
                                snippet: text,
                                url: topic.first_url,
                                source_type: "link".to_string(),
                            });
                        }
                    }
                }

                if hits.is_empty() {
                    hits.push(ExternalHit {
                        title: "No instant answer".to_string(),
                        snippet: "DuckDuckGo's Instant Answer API didn't return results for this query. Try a more specific term.".to_string(),
                        url: Some(format!("https://duckduckgo.com/?q={}", urlencoding(query))),
                        source_type: "link".to_string(),
                    });
                }

                ExternalResult::done("DuckDuckGo", hits)
            }
            Err(e) => ExternalResult::error("DuckDuckGo", &format!("Parse error: {}", e)),
        },
        Err(_) => ExternalResult::error(
            "DuckDuckGo",
            "CORS proxy not running. Start with: cargo run -p prism-proxy",
        ),
    }
}

// ═══════════════════════════════════════════════════════════════
// OLLAMA — Direct to user's localhost:11434 (graceful fallback)
// ═══════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct OllamaResponse {
    response: Option<String>,
    model: Option<String>,
}

pub async fn query_ollama(query: &str) -> ExternalResult {
    let url = "http://localhost:11434/api/generate";
    let body = serde_json::json!({
        "model": "gemma3:1b",
        "prompt": format!("Answer concisely in 2-3 sentences: {}", query),
        "stream": false,
    });

    let request = match gloo_net::http::Request::post(url)
        .header("Content-Type", "application/json")
        .body(body.to_string())
    {
        Ok(r) => r,
        Err(e) => return ExternalResult::error("Ollama", &format!("Request build error: {}", e)),
    };

    match request.send().await {
        Ok(resp) => match resp.json::<OllamaResponse>().await {
            Ok(data) => {
                let model = data.model.unwrap_or_else(|| "unknown".to_string());
                let text = data.response.unwrap_or_else(|| "No response".to_string());
                ExternalResult::done(
                    "Ollama",
                    vec![ExternalHit {
                        title: format!("AI Response ({})", model),
                        snippet: text,
                        url: None,
                        source_type: "ai-generated".to_string(),
                    }],
                )
            }
            Err(e) => ExternalResult::error("Ollama", &format!("Parse error: {}", e)),
        },
        Err(_) => ExternalResult::error(
            "Ollama",
            "Local Ollama not detected.\n\nTo enable AI comparison, start Ollama with CORS:\nOLLAMA_ORIGINS=\"*\" ollama serve",
        ),
    }
}

// ═══════════════════════════════════════════════════════════════
// BRAVE SEARCH — Via CORS proxy (requires API key in localStorage)
// ═══════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct BraveResponse {
    web: Option<BraveWebResults>,
}

#[derive(Deserialize)]
struct BraveWebResults {
    results: Option<Vec<BraveResult>>,
}

#[derive(Deserialize)]
struct BraveResult {
    title: Option<String>,
    description: Option<String>,
    url: Option<String>,
}

pub async fn search_brave(query: &str) -> ExternalResult {
    let api_key = get_local_storage_item("brave-api-key");

    let api_key = match api_key {
        Some(k) if !k.is_empty() => k,
        _ => {
            return ExternalResult::error(
                "Brave",
                "Brave Search requires an API key.\n\nGet one at https://api.search.brave.com\nThen add it in Settings.",
            );
        }
    };

    let proxy_base = get_proxy_base();
    let url = format!("{}/api/brave?q={}", proxy_base, urlencoding(query),);

    match gloo_net::http::Request::get(&url)
        .header("X-Brave-Key", &api_key)
        .send()
        .await
    {
        Ok(resp) => match resp.json::<BraveResponse>().await {
            Ok(data) => {
                let hits: Vec<ExternalHit> = data
                    .web
                    .and_then(|w| w.results)
                    .unwrap_or_default()
                    .into_iter()
                    .take(5)
                    .map(|r| ExternalHit {
                        title: r.title.unwrap_or_default(),
                        snippet: r.description.unwrap_or_default(),
                        url: r.url,
                        source_type: "link".to_string(),
                    })
                    .collect();

                if hits.is_empty() {
                    ExternalResult::done(
                        "Brave",
                        vec![ExternalHit {
                            title: "No results".to_string(),
                            snippet: "Brave Search returned no results for this query.".to_string(),
                            url: Some(format!(
                                "https://search.brave.com/search?q={}",
                                urlencoding(query)
                            )),
                            source_type: "link".to_string(),
                        }],
                    )
                } else {
                    ExternalResult::done("Brave", hits)
                }
            }
            Err(e) => ExternalResult::error("Brave", &format!("Parse error: {}", e)),
        },
        Err(_) => ExternalResult::error(
            "Brave",
            "CORS proxy not running. Start with: cargo run -p prism-proxy",
        ),
    }
}

// ═══════════════════════════════════════════════════════════════
// PERPLEXITY — Via CORS proxy (requires API key in localStorage)
// ═══════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct PerplexityResponse {
    choices: Option<Vec<PerplexityChoice>>,
    model: Option<String>,
}

#[derive(Deserialize)]
struct PerplexityChoice {
    message: Option<PerplexityMessage>,
}

#[derive(Deserialize)]
struct PerplexityMessage {
    content: Option<String>,
}

pub async fn query_perplexity(query: &str) -> ExternalResult {
    let api_key = get_local_storage_item("perplexity-api-key");

    let api_key = match api_key {
        Some(k) if !k.is_empty() => k,
        _ => {
            return ExternalResult::error(
                "Perplexity",
                "Perplexity requires an API key.\n\nGet one at https://docs.perplexity.ai\nThen add it in Settings.",
            );
        }
    };

    let proxy_base = get_proxy_base();
    let url = format!("{}/api/perplexity", proxy_base);

    let body = serde_json::json!({
        "model": "sonar",
        "messages": [
            { "role": "user", "content": query }
        ]
    });

    let request = match gloo_net::http::Request::post(&url)
        .header("Content-Type", "application/json")
        .header("X-Perplexity-Key", &api_key)
        .body(body.to_string())
    {
        Ok(r) => r,
        Err(e) => {
            return ExternalResult::error("Perplexity", &format!("Request build error: {}", e));
        }
    };

    match request.send().await {
        Ok(resp) => match resp.json::<PerplexityResponse>().await {
            Ok(data) => {
                let model = data.model.unwrap_or_else(|| "sonar".to_string());
                let text = data
                    .choices
                    .and_then(|c| c.into_iter().next())
                    .and_then(|c| c.message)
                    .and_then(|m| m.content)
                    .unwrap_or_else(|| "No response".to_string());

                ExternalResult::done(
                    "Perplexity",
                    vec![ExternalHit {
                        title: format!("AI Response ({})", model),
                        snippet: text,
                        url: None,
                        source_type: "ai-generated".to_string(),
                    }],
                )
            }
            Err(e) => ExternalResult::error("Perplexity", &format!("Parse error: {}", e)),
        },
        Err(_) => ExternalResult::error(
            "Perplexity",
            "CORS proxy not running. Start with: cargo run -p prism-proxy",
        ),
    }
}

// ═══════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════

fn urlencoding(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' ' => '+'.to_string(),
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            _ => format!("%{:02X}", c as u32),
        })
        .collect()
}

fn strip_html_tags(html: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }
    result
}

fn get_proxy_base() -> String {
    // Same-origin: proxy routes are now served by prism-serve on the same port.
    // No separate :8131 proxy needed.
    String::new()
}

fn get_local_storage_item(key: &str) -> Option<String> {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(key).ok().flatten())
}
