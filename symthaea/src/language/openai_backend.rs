// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! OpenAI LLM Backend
//!
//! Implements the `LLMBackend` trait for OpenAI's API (GPT-4o, GPT-4o-mini, etc).
//! Reads `OPENAI_API_KEY` from environment. Uses reqwest for HTTP calls.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::llm_backend::{GenerationParams, LLMBackend};

/// OpenAI backend connecting to the OpenAI API.
pub struct OpenAiBackend {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    max_tokens: usize,
    stream: bool,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
}

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Deserialize)]
struct StreamDelta {
    content: Option<String>,
}

impl OpenAiBackend {
    /// Create a new OpenAI backend.
    ///
    /// Reads `OPENAI_API_KEY` and optionally `OPENAI_MODEL` from environment.
    /// Returns `None` if `OPENAI_API_KEY` is not set.
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").ok()?;
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
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
impl LLMBackend for OpenAiBackend {
    async fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<String> {
        let mut messages = Vec::new();
        if let Some(ref sys) = params.system_prompt {
            messages.push(ChatMessage {
                role: "system",
                content: sys,
            });
        }
        messages.push(ChatMessage {
            role: "user",
            content: prompt,
        });

        let body = ChatRequest {
            model: &self.model,
            messages,
            temperature: params.temperature,
            max_tokens: params.max_tokens,
            stream: false,
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("OpenAI request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<body unreadable>".to_string());
            anyhow::bail!("OpenAI returned {status}: {text}");
        }

        let chat_resp: ChatResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse OpenAI response: {e}"))?;

        chat_resp
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("OpenAI returned empty response"))
    }

    async fn generate_streaming(
        &self,
        prompt: &str,
        params: &GenerationParams,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<String> {
        let mut messages = Vec::new();
        if let Some(ref sys) = params.system_prompt {
            messages.push(ChatMessage {
                role: "system",
                content: sys,
            });
        }
        messages.push(ChatMessage {
            role: "user",
            content: prompt,
        });

        let body = ChatRequest {
            model: &self.model,
            messages,
            temperature: params.temperature,
            max_tokens: params.max_tokens,
            stream: true,
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("OpenAI streaming request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<body unreadable>".to_string());
            anyhow::bail!("OpenAI returned {status}: {text}");
        }

        let mut full_response = String::new();
        let mut buffer = Vec::new();
        let mut stream = response;

        while let Some(chunk) = stream
            .chunk()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read OpenAI stream: {e}"))?
        {
            buffer.extend_from_slice(&chunk);

            while let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buffer.drain(..=newline_pos).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                let line = line.trim();

                if line.is_empty() || line == "data: [DONE]" {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(ref content) = choice.delta.content {
                                full_response.push_str(content);
                                on_token(content);
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
        "OpenAI"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_from_env_missing_key() {
        // Save and clear
        let prev = std::env::var("OPENAI_API_KEY").ok();
        // SAFETY: This test temporarily owns the process environment key and
        // restores it before returning.
        unsafe { std::env::remove_var("OPENAI_API_KEY") };

        assert!(OpenAiBackend::from_env().is_none());

        if let Some(val) = prev {
            // SAFETY: Restores the process environment value saved above.
            unsafe { std::env::set_var("OPENAI_API_KEY", val) };
        }
    }

    #[test]
    fn test_openai_creation() {
        let backend = OpenAiBackend::new(
            "test-key".to_string(),
            "gpt-4o-mini".to_string(),
            "https://api.openai.com/v1".to_string(),
        );
        assert_eq!(backend.name(), "OpenAI");
        assert_eq!(backend.model, "gpt-4o-mini");
    }

    #[tokio::test]
    async fn test_openai_is_available() {
        let backend = OpenAiBackend::new(
            "test-key".to_string(),
            "gpt-4o-mini".to_string(),
            "https://api.openai.com/v1".to_string(),
        );
        assert!(backend.is_available().await);

        let empty = OpenAiBackend::new(
            "".to_string(),
            "gpt-4o-mini".to_string(),
            "https://api.openai.com/v1".to_string(),
        );
        assert!(!empty.is_available().await);
    }
}
