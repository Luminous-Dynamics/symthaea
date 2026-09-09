// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generic OpenAI-compatible HTTP transport.
//!
//! This is transport only. Provider identity, privacy admission, credentials as
//! authority, routing, quota policy, and model trust remain outside this module.
//! A later integration layer will adapt this transport to `LLMBackend` only after
//! an inference route has been admitted.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;

/// Bearer credential with redacted `Debug` output.
#[derive(Clone, PartialEq, Eq)]
pub struct BearerCredential(String);

impl BearerCredential {
    pub fn new(value: impl Into<String>) -> Result<Self, TransportConfigError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(TransportConfigError::EmptyCredential);
        }
        Ok(Self(value))
    }

    fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for BearerCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BearerCredential([REDACTED])")
    }
}

/// Credential mode for an OpenAI-compatible endpoint.
#[derive(Clone, PartialEq, Eq)]
pub enum TransportCredential {
    None,
    Bearer(BearerCredential),
}

impl fmt::Debug for TransportCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("TransportCredential::None"),
            Self::Bearer(_) => f.write_str("TransportCredential::Bearer([REDACTED])"),
        }
    }
}

/// Immutable transport configuration.
#[derive(Clone)]
pub struct OpenAiCompatibleConfig {
    provider_id: String,
    base_url: reqwest::Url,
    model: String,
    credential: TransportCredential,
    timeout: Duration,
}

impl fmt::Debug for OpenAiCompatibleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiCompatibleConfig")
            .field("provider_id", &self.provider_id)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("credential", &self.credential)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl OpenAiCompatibleConfig {
    pub fn new(
        provider_id: impl Into<String>,
        base_url: &str,
        model: impl Into<String>,
        credential: TransportCredential,
    ) -> Result<Self, TransportConfigError> {
        let provider_id = provider_id.into();
        if provider_id.trim().is_empty() {
            return Err(TransportConfigError::EmptyProviderId);
        }

        let model = model.into();
        if model.trim().is_empty() {
            return Err(TransportConfigError::EmptyModel);
        }

        let mut base_url = reqwest::Url::parse(base_url)
            .map_err(|_| TransportConfigError::InvalidBaseUrl)?;
        if !matches!(base_url.scheme(), "http" | "https") {
            return Err(TransportConfigError::UnsupportedUrlScheme);
        }
        if base_url.cannot_be_a_base() {
            return Err(TransportConfigError::InvalidBaseUrl);
        }

        // Canonicalize to one trailing slash so Url::join has deterministic semantics.
        if !base_url.path().ends_with('/') {
            let mut path = base_url.path().to_owned();
            path.push('/');
            base_url.set_path(&path);
        }

        Ok(Self {
            provider_id,
            base_url,
            model,
            credential,
            timeout: Duration::from_secs(120),
        })
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self, TransportConfigError> {
        if timeout.is_zero() {
            return Err(TransportConfigError::ZeroTimeout);
        }
        self.timeout = timeout;
        Ok(self)
    }

    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn base_url(&self) -> &reqwest::Url {
        &self.base_url
    }

    fn chat_completions_url(&self) -> Result<reqwest::Url, TransportConfigError> {
        self.base_url
            .join("chat/completions")
            .map_err(|_| TransportConfigError::InvalidBaseUrl)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportConfigError {
    EmptyProviderId,
    EmptyModel,
    EmptyCredential,
    InvalidBaseUrl,
    UnsupportedUrlScheme,
    ZeroTimeout,
}

impl fmt::Display for TransportConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyProviderId => write!(f, "provider id must not be empty"),
            Self::EmptyModel => write!(f, "model id must not be empty"),
            Self::EmptyCredential => write!(f, "bearer credential must not be empty"),
            Self::InvalidBaseUrl => write!(f, "invalid base URL"),
            Self::UnsupportedUrlScheme => write!(f, "base URL must use http or https"),
            Self::ZeroTimeout => write!(f, "transport timeout must be non-zero"),
        }
    }
}

impl std::error::Error for TransportConfigError {}

/// Wire-level generation request.
#[derive(Debug, Clone)]
pub struct TransportGenerationRequest {
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub temperature: f32,
    pub max_tokens: usize,
}

/// Successful wire-level response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportGenerationResponse {
    pub text: String,
}

/// Transport failure deliberately excludes remote response bodies.
///
/// Provider error bodies can echo prompt/context. Keeping them out of this error
/// type prevents routine logging from becoming a secondary data-exfiltration path.
#[derive(Debug)]
pub enum TransportError {
    Config(TransportConfigError),
    ClientBuild,
    Request,
    HttpStatus(u16),
    ResponseDecode,
    EmptyResponse,
    StreamRead,
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(e) => write!(f, "transport configuration error: {e}"),
            Self::ClientBuild => write!(f, "failed to construct HTTP client"),
            Self::Request => write!(f, "OpenAI-compatible request failed"),
            Self::HttpStatus(status) => write!(f, "OpenAI-compatible endpoint returned HTTP {status}"),
            Self::ResponseDecode => write!(f, "failed to decode OpenAI-compatible response"),
            Self::EmptyResponse => write!(f, "OpenAI-compatible endpoint returned no text"),
            Self::StreamRead => write!(f, "failed while reading OpenAI-compatible stream"),
        }
    }
}

impl std::error::Error for TransportError {}

impl From<TransportConfigError> for TransportError {
    fn from(value: TransportConfigError) -> Self {
        Self::Config(value)
    }
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

/// Reusable OpenAI-compatible client. It contains no routing or admission logic.
pub struct OpenAiCompatibleTransport {
    config: OpenAiCompatibleConfig,
    client: reqwest::Client,
}

impl fmt::Debug for OpenAiCompatibleTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiCompatibleTransport")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl OpenAiCompatibleTransport {
    pub fn new(config: OpenAiCompatibleConfig) -> Result<Self, TransportError> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|_| TransportError::ClientBuild)?;
        Ok(Self { config, client })
    }

    pub fn config(&self) -> &OpenAiCompatibleConfig {
        &self.config
    }

    fn request_builder(&self, url: reqwest::Url) -> reqwest::RequestBuilder {
        let request = self.client.post(url);
        match &self.config.credential {
            TransportCredential::None => request,
            TransportCredential::Bearer(token) => request.bearer_auth(token.expose()),
        }
    }

    fn body<'a>(
        &'a self,
        request: &'a TransportGenerationRequest,
        stream: bool,
    ) -> ChatRequest<'a> {
        let mut messages = Vec::with_capacity(2);
        if let Some(system_prompt) = request.system_prompt.as_deref() {
            messages.push(ChatMessage {
                role: "system",
                content: system_prompt,
            });
        }
        messages.push(ChatMessage {
            role: "user",
            content: &request.prompt,
        });

        ChatRequest {
            model: &self.config.model,
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream,
        }
    }

    pub async fn generate(
        &self,
        request: &TransportGenerationRequest,
    ) -> Result<TransportGenerationResponse, TransportError> {
        let url = self.config.chat_completions_url()?;
        let response = self
            .request_builder(url)
            .json(&self.body(request, false))
            .send()
            .await
            .map_err(|_| TransportError::Request)?;

        if !response.status().is_success() {
            return Err(TransportError::HttpStatus(response.status().as_u16()));
        }

        let decoded: ChatResponse = response
            .json()
            .await
            .map_err(|_| TransportError::ResponseDecode)?;
        let text = decoded
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .filter(|text| !text.is_empty())
            .ok_or(TransportError::EmptyResponse)?;

        Ok(TransportGenerationResponse { text })
    }

    pub async fn generate_streaming(
        &self,
        request: &TransportGenerationRequest,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<TransportGenerationResponse, TransportError> {
        let url = self.config.chat_completions_url()?;
        let mut response = self
            .request_builder(url)
            .json(&self.body(request, true))
            .send()
            .await
            .map_err(|_| TransportError::Request)?;

        if !response.status().is_success() {
            return Err(TransportError::HttpStatus(response.status().as_u16()));
        }

        let mut full_response = String::new();
        let mut buffer = Vec::new();

        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| TransportError::StreamRead)?
        {
            buffer.extend_from_slice(&chunk);

            while let Some(newline_pos) = buffer.iter().position(|&byte| byte == b'\n') {
                let line_bytes: Vec<u8> = buffer.drain(..=newline_pos).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                let line = line.trim();
                if line.is_empty() || line == "data: [DONE]" {
                    continue;
                }
                if let Some(data) = line.strip_prefix("data: ")
                    && let Ok(chunk) = serde_json::from_str::<StreamChunk>(data)
                    && let Some(choice) = chunk.choices.first()
                    && let Some(content) = choice.delta.content.as_deref()
                {
                    full_response.push_str(content);
                    on_token(content);
                }
            }
        }

        if full_response.is_empty() {
            return Err(TransportError::EmptyResponse);
        }
        Ok(TransportGenerationResponse {
            text: full_response,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_debug_is_redacted() {
        let credential = BearerCredential::new("top-secret-token").unwrap();
        let debug = format!("{credential:?}");
        assert!(!debug.contains("top-secret-token"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn config_debug_is_redacted() {
        let config = OpenAiCompatibleConfig::new(
            "example",
            "https://example.test/v1",
            "model",
            TransportCredential::Bearer(BearerCredential::new("secret").unwrap()),
        )
        .unwrap();
        let debug = format!("{config:?}");
        assert!(!debug.contains("secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn endpoint_join_is_canonical() {
        for base in ["https://example.test/v1", "https://example.test/v1/"] {
            let config = OpenAiCompatibleConfig::new(
                "example",
                base,
                "model",
                TransportCredential::None,
            )
            .unwrap();
            assert_eq!(
                config.chat_completions_url().unwrap().as_str(),
                "https://example.test/v1/chat/completions"
            );
        }
    }

    #[test]
    fn invalid_or_non_http_endpoint_is_rejected() {
        assert!(matches!(
            OpenAiCompatibleConfig::new("p", "not a url", "m", TransportCredential::None),
            Err(TransportConfigError::InvalidBaseUrl)
        ));
        assert!(matches!(
            OpenAiCompatibleConfig::new("p", "file:///tmp/api", "m", TransportCredential::None),
            Err(TransportConfigError::UnsupportedUrlScheme)
        ));
    }

    #[test]
    fn zero_timeout_is_rejected() {
        let config = OpenAiCompatibleConfig::new(
            "example",
            "https://example.test/v1",
            "model",
            TransportCredential::None,
        )
        .unwrap();
        assert!(matches!(
            config.with_timeout(Duration::ZERO),
            Err(TransportConfigError::ZeroTimeout)
        ));
    }

    #[test]
    fn transport_error_does_not_have_a_remote_body_variant() {
        let rendered = TransportError::HttpStatus(429).to_string();
        assert_eq!(rendered, "OpenAI-compatible endpoint returned HTTP 429");
    }
}
