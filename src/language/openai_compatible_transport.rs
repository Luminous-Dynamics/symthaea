// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generic OpenAI-compatible HTTP transport.
//!
//! This is transport only. Provider identity, privacy admission, credentials as
//! authority, routing, quota policy, and model trust remain outside this module.
//!
//! IF-5 adds an *observed* API alongside the legacy response API. Observations
//! contain bounded provider-declared wire metadata and normalized rate-limit
//! headers, never provider error bodies or prompt content.

use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{Duration, Instant};

const MAX_WIRE_METADATA_BYTES: usize = 1024;

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

        let mut base_url =
            reqwest::Url::parse(base_url).map_err(|_| TransportConfigError::InvalidBaseUrl)?;
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

/// Provider-declared token usage. These values are observations, not independently
/// verified tokenizer counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TransportTokenUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

/// Normalized rate-limit hints observed on the HTTP response.
///
/// Reset fields are durations from observation time, never trusted wall-clock
/// timestamps. Unknown/unparseable provider formats remain `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TransportRateLimitObservation {
    pub retry_after_millis: Option<u64>,
    pub request_limit: Option<u64>,
    pub request_remaining: Option<u64>,
    pub request_reset_after_millis: Option<u64>,
    pub token_limit: Option<u64>,
    pub token_remaining: Option<u64>,
    pub token_reset_after_millis: Option<u64>,
}

/// Bounded wire evidence observed from one provider response.
///
/// All metadata fields are provider claims/identifiers. They are bounded and
/// control-character-free before being retained.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TransportWireObservation {
    pub response_id: Option<String>,
    pub provider_model: Option<String>,
    pub system_fingerprint: Option<String>,
    pub finish_reason: Option<String>,
    pub request_id_header: Option<String>,
    pub usage: Option<TransportTokenUsage>,
    pub rate_limits: TransportRateLimitObservation,
    pub latency_millis: u64,
    pub metadata_conflict: bool,
    pub metadata_rejected: bool,
}

/// Successful response paired with provider-observed wire evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedTransportGeneration {
    pub response: TransportGenerationResponse,
    pub observation: TransportWireObservation,
}

/// Failed transport attempt paired with whatever non-body evidence was safely
/// observable. The failure deliberately retains no remote response body.
#[derive(Debug)]
pub struct ObservedTransportFailure {
    pub error: TransportError,
    pub observation: TransportWireObservation,
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
            Self::HttpStatus(status) => {
                write!(f, "OpenAI-compatible endpoint returned HTTP {status}")
            }
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
    id: Option<String>,
    model: Option<String>,
    system_fingerprint: Option<String>,
    usage: Option<ChatUsage>,
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    finish_reason: Option<String>,
    message: ChatMessageResponse,
}

#[derive(Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct ChatUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

impl From<ChatUsage> for TransportTokenUsage {
    fn from(value: ChatUsage) -> Self {
        Self {
            prompt_tokens: value.prompt_tokens,
            completion_tokens: value.completion_tokens,
            total_tokens: value.total_tokens,
        }
    }
}

#[derive(Deserialize)]
struct StreamChunk {
    id: Option<String>,
    model: Option<String>,
    system_fingerprint: Option<String>,
    usage: Option<ChatUsage>,
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    finish_reason: Option<String>,
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

    /// Legacy response-only API. IF-5 callers should prefer `generate_observed`.
    pub async fn generate(
        &self,
        request: &TransportGenerationRequest,
    ) -> Result<TransportGenerationResponse, TransportError> {
        self.generate_observed(request)
            .await
            .map(|observed| observed.response)
            .map_err(|failure| failure.error)
    }

    /// Execute one non-streaming request and retain bounded, non-body wire evidence.
    pub async fn generate_observed(
        &self,
        request: &TransportGenerationRequest,
    ) -> Result<ObservedTransportGeneration, ObservedTransportFailure> {
        let started = Instant::now();
        let url = self.config.chat_completions_url().map_err(|error| {
            observed_failure(TransportError::Config(error), started, HeaderMap::new())
        })?;

        let response = self
            .request_builder(url)
            .json(&self.body(request, false))
            .send()
            .await
            .map_err(|_| observed_failure(TransportError::Request, started, HeaderMap::new()))?;

        let status = response.status();
        let headers = response.headers().clone();
        let mut observation = observation_from_headers(&headers);
        if !status.is_success() {
            observation.latency_millis = elapsed_millis(started);
            return Err(ObservedTransportFailure {
                error: TransportError::HttpStatus(status.as_u16()),
                observation,
            });
        }

        let decoded: ChatResponse = response.json().await.map_err(|_| {
            observation.latency_millis = elapsed_millis(started);
            ObservedTransportFailure {
                error: TransportError::ResponseDecode,
                observation: observation.clone(),
            }
        })?;

        observe_metadata(
            &mut observation.response_id,
            decoded.id.as_deref(),
            &mut observation.metadata_conflict,
            &mut observation.metadata_rejected,
        );
        observe_metadata(
            &mut observation.provider_model,
            decoded.model.as_deref(),
            &mut observation.metadata_conflict,
            &mut observation.metadata_rejected,
        );
        observe_metadata(
            &mut observation.system_fingerprint,
            decoded.system_fingerprint.as_deref(),
            &mut observation.metadata_conflict,
            &mut observation.metadata_rejected,
        );
        if let Some(choice) = decoded.choices.first() {
            observe_metadata(
                &mut observation.finish_reason,
                choice.finish_reason.as_deref(),
                &mut observation.metadata_conflict,
                &mut observation.metadata_rejected,
            );
        }
        observation.usage = decoded.usage.map(Into::into);

        let text = decoded
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .filter(|text| !text.is_empty())
            .ok_or_else(|| {
                observation.latency_millis = elapsed_millis(started);
                ObservedTransportFailure {
                    error: TransportError::EmptyResponse,
                    observation: observation.clone(),
                }
            })?;

        observation.latency_millis = elapsed_millis(started);
        Ok(ObservedTransportGeneration {
            response: TransportGenerationResponse { text },
            observation,
        })
    }

    /// Legacy streaming API. IF-5 callers should prefer `generate_streaming_observed`.
    pub async fn generate_streaming(
        &self,
        request: &TransportGenerationRequest,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<TransportGenerationResponse, TransportError> {
        self.generate_streaming_observed(request, on_token)
            .await
            .map(|observed| observed.response)
            .map_err(|failure| failure.error)
    }

    /// Stream response text while accumulating bounded provider-declared metadata.
    pub async fn generate_streaming_observed(
        &self,
        request: &TransportGenerationRequest,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<ObservedTransportGeneration, ObservedTransportFailure> {
        let started = Instant::now();
        let url = self.config.chat_completions_url().map_err(|error| {
            observed_failure(TransportError::Config(error), started, HeaderMap::new())
        })?;
        let mut response = self
            .request_builder(url)
            .json(&self.body(request, true))
            .send()
            .await
            .map_err(|_| observed_failure(TransportError::Request, started, HeaderMap::new()))?;

        let status = response.status();
        let headers = response.headers().clone();
        let mut observation = observation_from_headers(&headers);
        if !status.is_success() {
            observation.latency_millis = elapsed_millis(started);
            return Err(ObservedTransportFailure {
                error: TransportError::HttpStatus(status.as_u16()),
                observation,
            });
        }

        let mut full_response = String::new();
        let mut buffer = Vec::new();

        while let Some(chunk) = response.chunk().await.map_err(|_| {
            observation.latency_millis = elapsed_millis(started);
            ObservedTransportFailure {
                error: TransportError::StreamRead,
                observation: observation.clone(),
            }
        })? {
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
                {
                    observe_stream_chunk(&mut observation, &chunk);
                    if let Some(choice) = chunk.choices.first()
                        && let Some(content) = choice.delta.content.as_deref()
                    {
                        full_response.push_str(content);
                        on_token(content);
                    }
                }
            }
        }

        if full_response.is_empty() {
            observation.latency_millis = elapsed_millis(started);
            return Err(ObservedTransportFailure {
                error: TransportError::EmptyResponse,
                observation,
            });
        }

        observation.latency_millis = elapsed_millis(started);
        Ok(ObservedTransportGeneration {
            response: TransportGenerationResponse {
                text: full_response,
            },
            observation,
        })
    }
}

fn observed_failure(
    error: TransportError,
    started: Instant,
    headers: HeaderMap,
) -> ObservedTransportFailure {
    let mut observation = observation_from_headers(&headers);
    observation.latency_millis = elapsed_millis(started);
    ObservedTransportFailure { error, observation }
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn observation_from_headers(headers: &HeaderMap) -> TransportWireObservation {
    let mut observation = TransportWireObservation {
        rate_limits: TransportRateLimitObservation {
            retry_after_millis: header_value(headers, "retry-after")
                .and_then(parse_retry_after_millis),
            request_limit: header_u64_first(
                headers,
                &["x-ratelimit-limit-requests", "ratelimit-limit"],
            ),
            request_remaining: header_u64_first(
                headers,
                &["x-ratelimit-remaining-requests", "ratelimit-remaining"],
            ),
            request_reset_after_millis: header_value_first(
                headers,
                &["x-ratelimit-reset-requests", "ratelimit-reset"],
            )
            .and_then(parse_duration_millis),
            token_limit: header_u64_first(headers, &["x-ratelimit-limit-tokens"]),
            token_remaining: header_u64_first(headers, &["x-ratelimit-remaining-tokens"]),
            token_reset_after_millis: header_value_first(
                headers,
                &["x-ratelimit-reset-tokens"],
            )
            .and_then(parse_duration_millis),
        },
        ..TransportWireObservation::default()
    };

    let request_id = header_value_first(headers, &["x-request-id", "request-id"]);
    observe_metadata(
        &mut observation.request_id_header,
        request_id,
        &mut observation.metadata_conflict,
        &mut observation.metadata_rejected,
    );
    observation
}

fn observe_stream_chunk(observation: &mut TransportWireObservation, chunk: &StreamChunk) {
    observe_metadata(
        &mut observation.response_id,
        chunk.id.as_deref(),
        &mut observation.metadata_conflict,
        &mut observation.metadata_rejected,
    );
    observe_metadata(
        &mut observation.provider_model,
        chunk.model.as_deref(),
        &mut observation.metadata_conflict,
        &mut observation.metadata_rejected,
    );
    observe_metadata(
        &mut observation.system_fingerprint,
        chunk.system_fingerprint.as_deref(),
        &mut observation.metadata_conflict,
        &mut observation.metadata_rejected,
    );
    if let Some(choice) = chunk.choices.first() {
        observe_metadata(
            &mut observation.finish_reason,
            choice.finish_reason.as_deref(),
            &mut observation.metadata_conflict,
            &mut observation.metadata_rejected,
        );
    }
    if let Some(usage) = chunk.usage {
        let usage = TransportTokenUsage::from(usage);
        if observation.usage.is_some_and(|existing| existing != usage) {
            observation.metadata_conflict = true;
        } else {
            observation.usage = Some(usage);
        }
    }
}

fn observe_metadata(
    slot: &mut Option<String>,
    incoming: Option<&str>,
    conflict: &mut bool,
    rejected: &mut bool,
) {
    let Some(incoming) = incoming else {
        return;
    };
    let Some(incoming) = bounded_metadata(incoming) else {
        *rejected = true;
        return;
    };

    match slot {
        None => *slot = Some(incoming),
        Some(existing) if existing == &incoming => {}
        Some(_) => *conflict = true,
    }
}

fn bounded_metadata(value: &str) -> Option<String> {
    if value.is_empty()
        || value.len() > MAX_WIRE_METADATA_BYTES
        || value.chars().any(char::is_control)
    {
        return None;
    }
    Some(value.to_string())
}

fn header_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name)?.to_str().ok().map(str::trim)
}

fn header_value_first<'a>(headers: &'a HeaderMap, names: &[&str]) -> Option<&'a str> {
    names.iter().find_map(|name| header_value(headers, name))
}

fn header_u64_first(headers: &HeaderMap, names: &[&str]) -> Option<u64> {
    header_value_first(headers, names)?.parse().ok()
}

fn parse_retry_after_millis(value: &str) -> Option<u64> {
    parse_decimal_with_multiplier(value.trim(), 1_000)
}

fn parse_duration_millis(value: &str) -> Option<u64> {
    let value = value.trim();
    if value.is_empty() || !value.is_ascii() {
        return None;
    }

    // A bare number is interpreted as seconds for compatibility with common
    // rate-limit headers. Compound forms such as "2m59.56s" are also accepted.
    if !value.chars().any(|c| c.is_ascii_alphabetic()) {
        return parse_decimal_with_multiplier(value, 1_000);
    }

    let bytes = value.as_bytes();
    let mut index = 0usize;
    let mut total = 0u128;
    while index < bytes.len() {
        let number_start = index;
        let mut seen_digit = false;
        let mut seen_dot = false;
        while index < bytes.len() {
            let byte = bytes[index];
            if byte.is_ascii_digit() {
                seen_digit = true;
                index += 1;
            } else if byte == b'.' && !seen_dot {
                seen_dot = true;
                index += 1;
            } else {
                break;
            }
        }
        if !seen_digit || number_start == index {
            return None;
        }
        let number = &value[number_start..index];

        let multiplier = if value[index..].starts_with("ms") {
            index += 2;
            1u64
        } else if value[index..].starts_with('s') {
            index += 1;
            1_000u64
        } else if value[index..].starts_with('m') {
            index += 1;
            60_000u64
        } else if value[index..].starts_with('h') {
            index += 1;
            3_600_000u64
        } else {
            return None;
        };

        total = total.checked_add(u128::from(parse_decimal_with_multiplier(
            number, multiplier,
        )?))?;
        if total > u128::from(u64::MAX) {
            return None;
        }
    }

    u64::try_from(total).ok()
}

fn parse_decimal_with_multiplier(value: &str, multiplier: u64) -> Option<u64> {
    let (whole, fraction) = match value.split_once('.') {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (value, None),
    };
    if whole.is_empty() || !whole.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let whole: u128 = whole.parse().ok()?;
    let mut total = whole.checked_mul(u128::from(multiplier))?;

    if let Some(fraction) = fraction {
        if fraction.is_empty()
            || fraction.len() > 9
            || !fraction.bytes().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        let numerator: u128 = fraction.parse().ok()?;
        let denominator = 10u128.checked_pow(u32::try_from(fraction.len()).ok()?)?;
        let fractional = numerator
            .checked_mul(u128::from(multiplier))?
            .checked_div(denominator)?;
        total = total.checked_add(fractional)?;
    }

    u64::try_from(total).ok()
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

    #[test]
    fn groq_style_reset_durations_parse_exactly() {
        assert_eq!(parse_duration_millis("2m59.56s"), Some(179_560));
        assert_eq!(parse_duration_millis("7.66s"), Some(7_660));
        assert_eq!(parse_retry_after_millis("2"), Some(2_000));
    }

    #[test]
    fn wire_metadata_is_bounded_and_control_free() {
        assert_eq!(bounded_metadata("model/v1").as_deref(), Some("model/v1"));
        assert!(bounded_metadata("bad\nmetadata").is_none());
        assert!(bounded_metadata(&"x".repeat(MAX_WIRE_METADATA_BYTES + 1)).is_none());
    }
}
