// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Anthropic LLM Backend
//!
//! Implements the `LLMBackend` trait for Anthropic's Messages API.
//! Reads `ANTHROPIC_API_KEY` from environment. Uses reqwest for HTTP calls.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::llm_backend::{GenerationParams, LLMBackend};

/// Anthropic backend connecting to the Messages API.
pub struct AnthropicBackend {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: Vec<Message<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
}

#[derive(Deserialize)]
struct StreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    delta: Option<StreamDelta>,
}

#[derive(Deserialize)]
struct StreamDelta {
    #[serde(rename = "type")]
    _delta_type: Option<String>,
    text: Option<String>,
}

impl AnthropicBackend {
    /// Create from environment variables.
    ///
    /// Reads `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_MODEL`.
    /// Returns `None` if API key is not set.
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").ok()?;
        let model = std::env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());
        let base_url = std::env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
        Some(Self::new(api_key, model, base_url))
    }

    /// Create with explicit parameters.
    pub fn new(api_key: String, model: String, base_url: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_default();
        Self {
            api_key,
            model,
            base_url,
            client,
        }
    }
}

#[async_trait::async_trait]
impl LLMBackend for AnthropicBackend {
    async fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<String> {
        let body = MessagesRequest {
            model: &self.model,
            max_tokens: params.max_tokens,
            system: params.system_prompt.as_deref(),
            messages: vec![Message {
                role: "user",
                content: prompt,
            }],
            temperature: Some(params.temperature),
            stream: false,
        };

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Anthropic request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<body unreadable>".to_string());
            anyhow::bail!("Anthropic returned {status}: {text}");
        }

        let msg_resp: MessagesResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse Anthropic response: {e}"))?;

        let text = msg_resp
            .content
            .iter()
            .filter(|b| b.block_type == "text")
            .filter_map(|b| b.text.as_ref())
            .cloned()
            .collect::<Vec<_>>()
            .join("");

        if text.is_empty() {
            anyhow::bail!("Anthropic returned empty response");
        }
        Ok(text)
    }

    async fn generate_streaming(
        &self,
        prompt: &str,
        params: &GenerationParams,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<String> {
        let body = MessagesRequest {
            model: &self.model,
            max_tokens: params.max_tokens,
            system: params.system_prompt.as_deref(),
            messages: vec![Message {
                role: "user",
                content: prompt,
            }],
            temperature: Some(params.temperature),
            stream: true,
        };

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Anthropic streaming request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<body unreadable>".to_string());
            anyhow::bail!("Anthropic returned {status}: {text}");
        }

        let mut full_response = String::new();
        let mut buffer = Vec::new();
        let mut stream = response;

        while let Some(chunk) = stream
            .chunk()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read Anthropic stream: {e}"))?
        {
            buffer.extend_from_slice(&chunk);

            while let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buffer.drain(..=newline_pos).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                let line = line.trim();

                if line.is_empty() {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(event) = serde_json::from_str::<StreamEvent>(data) {
                        if event.event_type == "content_block_delta" {
                            if let Some(ref delta) = event.delta {
                                if let Some(ref text) = delta.text {
                                    full_response.push_str(text);
                                    on_token(text);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(full_response)
    }

    async fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    fn name(&self) -> &str {
        "Anthropic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_from_env_missing_key() {
        let prev = std::env::var("ANTHROPIC_API_KEY").ok();
        // SAFETY: This test temporarily owns the process environment key and
        // restores it before returning.
        unsafe { std::env::remove_var("ANTHROPIC_API_KEY") };

        assert!(AnthropicBackend::from_env().is_none());

        if let Some(val) = prev {
            // SAFETY: Restores the process environment value saved above.
            unsafe { std::env::set_var("ANTHROPIC_API_KEY", val) };
        }
    }

    #[test]
    fn test_anthropic_creation() {
        let backend = AnthropicBackend::new(
            "test-key".to_string(),
            "claude-sonnet-4-20250514".to_string(),
            "https://api.anthropic.com".to_string(),
        );
        assert_eq!(backend.name(), "Anthropic");
    }

    #[tokio::test]
    async fn test_anthropic_is_available() {
        let backend = AnthropicBackend::new(
            "test-key".to_string(),
            "claude-sonnet-4-20250514".to_string(),
            "https://api.anthropic.com".to_string(),
        );
        assert!(backend.is_available().await);

        let empty = AnthropicBackend::new(
            "".to_string(),
            "claude-sonnet-4-20250514".to_string(),
            "https://api.anthropic.com".to_string(),
        );
        assert!(!empty.is_available().await);
    }
}
