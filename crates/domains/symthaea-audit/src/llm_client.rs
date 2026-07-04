// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Minimal blocking chat-completion clients for Ollama, Anthropic, and OpenAI-compatible
//! endpoints.
//!
//! **Deliberate duplication, not an oversight.** The main `symthaea` crate already has a
//! full pluggable `LLMBackend` trait with real Anthropic/OpenAI/Ollama implementations
//! at `symthaea/src/language/{llm_backend.rs,anthropic_backend.rs,openai_backend.rs}`.
//! This module re-implements a deliberately smaller subset of that — one blocking
//! `complete(system, history) -> String` call, no streaming — over `ureq` instead of
//! `reqwest`/tokio, so this crate can stay independent of the main crate and its async
//! runtime. If the upstream provider request/response shapes drift, this module must be
//! updated independently; nothing enforces that automatically. A future cleanup could
//! extract a shared micro-crate once a third consumer of this pattern appears — out of
//! scope for now.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::cli::{Cli, LlmProviderArg};

// Generous on purpose: local Ollama models under real-world load (shared dev boxes,
// concurrent builds) can take minutes for a single turn, especially as the growing
// tool-call history lengthens the prompt each turn.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const DEFAULT_MAX_TOKENS: usize = 4096;
const DEFAULT_TEMPERATURE: f32 = 0.2;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Clone, Debug)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

/// A blocking chat-completion backend: one system prompt, a turn history, one response.
pub trait LlmClient {
    fn complete(&self, system_prompt: &str, history: &[Turn]) -> Result<String>;
    fn name(&self) -> &'static str;
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().timeout(REQUEST_TIMEOUT).build()
}

fn error_body(err: ureq::Error) -> String {
    match err {
        ureq::Error::Status(code, response) => {
            let body = response
                .into_string()
                .unwrap_or_else(|_| "<body unreadable>".to_string());
            format!("HTTP {code}: {body}")
        }
        ureq::Error::Transport(t) => format!("transport error: {t}"),
    }
}

// ---------------------------------------------------------------------------
// Ollama
// ---------------------------------------------------------------------------

pub struct OllamaClient {
    base_url: String,
    model: String,
}

impl OllamaClient {
    pub fn new(base_url: String, model: String) -> Self {
        Self { base_url, model }
    }
}

#[derive(Serialize)]
struct OllamaMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: usize,
}

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaMessage<'a>>,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatResponseMessage,
}

#[derive(Deserialize)]
struct OllamaChatResponseMessage {
    content: String,
}

impl LlmClient for OllamaClient {
    fn complete(&self, system_prompt: &str, history: &[Turn]) -> Result<String> {
        let mut messages = vec![OllamaMessage {
            role: "system",
            content: system_prompt,
        }];
        for turn in history {
            messages.push(OllamaMessage {
                role: match turn.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: &turn.content,
            });
        }
        let body = OllamaChatRequest {
            model: &self.model,
            messages,
            stream: false,
            options: OllamaOptions {
                temperature: DEFAULT_TEMPERATURE,
                num_predict: DEFAULT_MAX_TOKENS,
            },
        };
        let url = format!("{}/api/chat", self.base_url);
        let response = agent()
            .post(&url)
            .send_json(&body)
            .map_err(|e| anyhow::anyhow!("Ollama request failed: {}", error_body(e)))?;
        let parsed: OllamaChatResponse = response
            .into_json()
            .context("failed to parse Ollama response")?;
        Ok(parsed.message.content)
    }

    fn name(&self) -> &'static str {
        "ollama"
    }
}

// ---------------------------------------------------------------------------
// Anthropic
// ---------------------------------------------------------------------------

pub struct AnthropicClient {
    api_key: String,
    model: String,
    base_url: String,
}

impl AnthropicClient {
    pub fn new(api_key: String, model: String, base_url: String) -> Self {
        Self {
            api_key,
            model,
            base_url,
        }
    }
}

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: usize,
    system: &'a str,
    messages: Vec<AnthropicMessage<'a>>,
    temperature: f32,
    stream: bool,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
}

#[derive(Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
}

impl LlmClient for AnthropicClient {
    fn complete(&self, system_prompt: &str, history: &[Turn]) -> Result<String> {
        let messages: Vec<AnthropicMessage> = history
            .iter()
            .map(|t| AnthropicMessage {
                role: match t.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: &t.content,
            })
            .collect();
        let body = AnthropicRequest {
            model: &self.model,
            max_tokens: DEFAULT_MAX_TOKENS,
            system: system_prompt,
            messages,
            temperature: DEFAULT_TEMPERATURE,
            stream: false,
        };
        let url = format!("{}/v1/messages", self.base_url);
        let response = agent()
            .post(&url)
            .set("x-api-key", &self.api_key)
            .set("anthropic-version", "2023-06-01")
            .set("content-type", "application/json")
            .send_json(&body)
            .map_err(|e| anyhow::anyhow!("Anthropic request failed: {}", error_body(e)))?;
        let parsed: AnthropicResponse = response
            .into_json()
            .context("failed to parse Anthropic response")?;
        let text = parsed
            .content
            .iter()
            .filter(|b| b.block_type == "text")
            .filter_map(|b| b.text.as_deref())
            .collect::<Vec<_>>()
            .join("");
        if text.is_empty() {
            bail!("Anthropic returned an empty response");
        }
        Ok(text)
    }

    fn name(&self) -> &'static str {
        "anthropic"
    }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible
// ---------------------------------------------------------------------------

pub struct OpenAiClient {
    api_key: String,
    model: String,
    base_url: String,
}

impl OpenAiClient {
    pub fn new(api_key: String, model: String, base_url: String) -> Self {
        Self {
            api_key,
            model,
            base_url,
        }
    }
}

#[derive(Serialize)]
struct OpenAiMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct OpenAiRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAiMessage<'a>>,
    temperature: f32,
    max_tokens: usize,
    stream: bool,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(Deserialize)]
struct OpenAiResponseMessage {
    content: Option<String>,
}

impl LlmClient for OpenAiClient {
    fn complete(&self, system_prompt: &str, history: &[Turn]) -> Result<String> {
        let mut messages = vec![OpenAiMessage {
            role: "system",
            content: system_prompt,
        }];
        for turn in history {
            messages.push(OpenAiMessage {
                role: match turn.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: &turn.content,
            });
        }
        let body = OpenAiRequest {
            model: &self.model,
            messages,
            temperature: DEFAULT_TEMPERATURE,
            max_tokens: DEFAULT_MAX_TOKENS,
            stream: false,
        };
        let url = format!("{}/chat/completions", self.base_url);
        let response = agent()
            .post(&url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .send_json(&body)
            .map_err(|e| anyhow::anyhow!("OpenAI request failed: {}", error_body(e)))?;
        let parsed: OpenAiResponse = response
            .into_json()
            .context("failed to parse OpenAI response")?;
        parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("OpenAI returned an empty response"))
    }

    fn name(&self) -> &'static str {
        "openai"
    }
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// Priority: explicit `--llm-provider` flag > `SYMTHAEA_AUDIT_LLM_PROVIDER` env var >
/// `ANTHROPIC_API_KEY` present > `OPENAI_API_KEY` present > Ollama fallback.
/// Mirrors the spirit and priority order of `llm_backend.rs`'s `create_backend_from_env`,
/// scoped to this tool's own env var name.
pub fn detect_backend(cli: &Cli) -> Result<Box<dyn LlmClient>> {
    let choice = match cli.llm_provider {
        LlmProviderArg::Ollama => "ollama".to_string(),
        LlmProviderArg::Anthropic => "anthropic".to_string(),
        LlmProviderArg::Openai => "openai".to_string(),
        LlmProviderArg::Auto => detect_provider_name(),
    };

    match choice.as_str() {
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .context("--llm-provider anthropic requires ANTHROPIC_API_KEY")?;
            let model = cli
                .model
                .clone()
                .or_else(|| std::env::var("ANTHROPIC_MODEL").ok())
                .unwrap_or_else(|| "claude-sonnet-4-20250514".to_string());
            let base_url = std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
            Ok(Box::new(AnthropicClient::new(api_key, model, base_url)))
        }
        "openai" => {
            let api_key = std::env::var("OPENAI_API_KEY")
                .context("--llm-provider openai requires OPENAI_API_KEY")?;
            let model = cli
                .model
                .clone()
                .or_else(|| std::env::var("OPENAI_MODEL").ok())
                .unwrap_or_else(|| "gpt-4o-mini".to_string());
            let base_url = cli
                .openai_base_url
                .clone()
                .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
            Ok(Box::new(OpenAiClient::new(api_key, model, base_url)))
        }
        _ => {
            let model = cli
                .model
                .clone()
                .or_else(|| std::env::var("OLLAMA_MODEL").ok())
                .unwrap_or_else(|| "gemma4:e2b".to_string());
            let base_url = cli
                .ollama_base_url
                .clone()
                .or_else(|| std::env::var("OLLAMA_BASE_URL").ok())
                .unwrap_or_else(|| "http://localhost:11434".to_string());
            Ok(Box::new(OllamaClient::new(base_url, model)))
        }
    }
}

fn detect_provider_name() -> String {
    if let Ok(p) = std::env::var("SYMTHAEA_AUDIT_LLM_PROVIDER") {
        return p;
    }
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        return "anthropic".to_string();
    }
    if std::env::var("OPENAI_API_KEY").is_ok() {
        return "openai".to_string();
    }
    "ollama".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::path::PathBuf;

    fn base_cli() -> Cli {
        Cli {
            target: PathBuf::from("."),
            focus: None,
            out: None,
            llm_provider: LlmProviderArg::Auto,
            model: None,
            ollama_base_url: None,
            openai_base_url: None,
            max_turns: 40,
            allow_exec: None,
            single_shot_paths: None,
            verify: false,
            no_hints: false,
        }
    }

    fn clear_env() {
        for var in [
            "SYMTHAEA_AUDIT_LLM_PROVIDER",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
        ] {
            unsafe { std::env::remove_var(var) };
        }
    }

    #[test]
    #[serial]
    fn auto_falls_back_to_ollama_with_nothing_set() {
        clear_env();
        let client = detect_backend(&base_cli()).unwrap();
        assert_eq!(client.name(), "ollama");
    }

    #[test]
    #[serial]
    fn auto_picks_anthropic_when_key_present() {
        clear_env();
        unsafe { std::env::set_var("ANTHROPIC_API_KEY", "test-key") };
        let client = detect_backend(&base_cli()).unwrap();
        assert_eq!(client.name(), "anthropic");
        clear_env();
    }

    #[test]
    #[serial]
    fn auto_picks_openai_when_only_openai_key_present() {
        clear_env();
        unsafe { std::env::set_var("OPENAI_API_KEY", "test-key") };
        let client = detect_backend(&base_cli()).unwrap();
        assert_eq!(client.name(), "openai");
        clear_env();
    }

    #[test]
    #[serial]
    fn explicit_flag_overrides_env() {
        clear_env();
        unsafe { std::env::set_var("ANTHROPIC_API_KEY", "test-key") };
        let mut cli = base_cli();
        cli.llm_provider = LlmProviderArg::Ollama;
        let client = detect_backend(&cli).unwrap();
        assert_eq!(client.name(), "ollama");
        clear_env();
    }

    #[test]
    #[serial]
    fn explicit_anthropic_without_key_errors() {
        clear_env();
        let mut cli = base_cli();
        cli.llm_provider = LlmProviderArg::Anthropic;
        let result = detect_backend(&cli);
        assert!(result.is_err());
    }
}
