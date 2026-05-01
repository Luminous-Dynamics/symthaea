// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # LLM Backend: Trait and Implementations
//!
//! Provides the abstraction for connecting to language model backends.
//! The primary implementation is `OllamaBackend` which connects to a local
//! Ollama instance. A `SimulatedBackend` provides fallback when no LLM is available.
//!
//! Design principle: "HDC+LTC THINKS. LLM TRANSLATES."
//! The LLM is used purely for natural language translation of structured thoughts,
//! not for reasoning or decision-making.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Parameters for LLM generation.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// Temperature for sampling (0.0 = deterministic, 1.0 = creative).
    pub temperature: f32,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// System prompt to set context.
    pub system_prompt: Option<String>,
    /// Optional consciousness context for consciousness-aware generation.
    /// When present, OllamaBackend injects this into the system prompt
    /// and adjusts temperature based on type_confidence.
    pub consciousness_context: Option<ConsciousnessContext>,
}

/// Consciousness context carried through to LLM backends.
///
/// Enables consciousness-aware code generation: the LLM receives
/// epistemic status, code quality signals, and consciousness level
/// as structured context rather than relying on prompt engineering alone.
#[derive(Debug, Clone)]
pub struct ConsciousnessContext {
    /// Epistemic status as string (Certain, Probable, Uncertain, Unknown, OutOfDomain)
    pub epistemic_status: String,
    /// Consciousness level (Phi, 0.0-1.0)
    pub phi: f32,
    /// Code-specific: type confidence (0.0-1.0, 0 = unknown types)
    pub type_confidence: f32,
    /// Code-specific: algorithm pattern strength (0.0-1.0)
    pub algorithm_pattern: f32,
    /// Code-specific: error likelihood (0.0-1.0)
    pub error_likelihood: f32,
    /// Code-specific: syntax complexity (0.0-1.0)
    pub syntax_complexity: f32,
}

impl ConsciousnessContext {
    /// Format as a system prompt supplement for code generation.
    pub fn to_system_supplement(&self) -> String {
        let mut parts = Vec::new();
        parts.push(format!("EPISTEMIC_STATUS: {}", self.epistemic_status));
        parts.push(format!("CONSCIOUSNESS_LEVEL: {:.2}", self.phi));

        if self.type_confidence > 0.0 || self.algorithm_pattern > 0.0 {
            parts.push(format!("TYPE_CONFIDENCE: {:.2}", self.type_confidence));
            parts.push(format!("ALGORITHM_PATTERN: {:.2}", self.algorithm_pattern));
            parts.push(format!("ERROR_LIKELIHOOD: {:.2}", self.error_likelihood));
            parts.push(format!("SYNTAX_COMPLEXITY: {:.2}", self.syntax_complexity));
        }

        if self.error_likelihood > 0.5 {
            parts.push(
                "NOTE: High error likelihood — add extra error handling and validation."
                    .to_string(),
            );
        }
        if self.type_confidence < 0.3 {
            parts.push("NOTE: Low type confidence — prefer explicit type annotations.".to_string());
        }
        if self.epistemic_status == "Uncertain" || self.epistemic_status == "Unknown" {
            parts.push(
                "NOTE: Low epistemic confidence — prefer the simplest well-typed implementation and explicit validation."
                    .to_string(),
            );
        }

        parts.join("\n")
    }

    /// Suggest temperature adjustment based on consciousness context.
    /// High type_confidence → lower temperature (more deterministic).
    /// High error_likelihood → lower temperature (more cautious).
    pub fn temperature_adjustment(&self) -> f32 {
        let confidence_factor = self.type_confidence * 0.2; // up to -0.2
        let error_factor = self.error_likelihood * 0.15; // up to -0.15
        -(confidence_factor + error_factor) // negative = cooler
    }
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            max_tokens: 256,
            system_prompt: None,
            consciousness_context: None,
        }
    }
}

/// Trait for LLM backend implementations.
///
/// Backends handle the actual text generation, whether via a local model
/// (Ollama), a remote API, or simulation.
#[async_trait::async_trait]
pub trait LLMBackend: Send + Sync {
    /// Generate a response from a prompt.
    async fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<String>;

    /// Update the affective state (thermodynamics -> language warping).
    fn update_affect(&self, _load: f32, _temp: f32) {}

    /// Apply a LoRA adapter to the model (v1.7.0 Broca Phase).
    fn apply_lora(&self, _lora_id: &str, _delta: &[u8]) -> Result<()> {
        Ok(())
    }

    /// Check if the backend is currently available.
    async fn is_available(&self) -> bool;

    /// Generate a streaming response, calling `on_token` for each token as it arrives.
    ///
    /// Returns the full concatenated response. The default implementation falls back
    /// to the non-streaming `generate` method.
    async fn generate_streaming(
        &self,
        prompt: &str,
        params: &GenerationParams,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<String> {
        let result = self.generate(prompt, params).await?;
        let cloned = result.clone();
        on_token(&cloned);
        Ok(result)
    }

    /// Generate from pre-computed thought channels (Direct Neural Path).
    /// Returns `None` if this backend doesn't support direct channel input.
    #[cfg(feature = "ssm_language")]
    fn generate_from_channels_direct(
        &self,
        _channels: &symthaea_broca::ThoughtChannels,
        _params: &GenerationParams,
    ) -> Option<Result<String>> {
        None
    }

    /// Return the last L-SSM semantic prediction error (0.0 if not applicable).
    #[cfg(feature = "liquid-mamba")]
    fn last_semantic_pe(&self) -> f32 {
        0.0
    }

    /// Export projection weights for federated swarm exchange.
    /// Returns `None` if this backend doesn't support swarm gradient exchange.
    #[cfg(feature = "liquid-mamba")]
    fn export_gradient(
        &self,
        _source_id: [u8; 32],
        _trust: f32,
        _version: u64,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Apply aggregated projection weights from swarm peers.
    /// Returns `true` if weights were successfully applied.
    #[cfg(feature = "liquid-mamba")]
    fn apply_aggregated_gradient(&self, _weights: &[f32]) -> bool {
        false
    }

    /// Pass FEP learning signal to modulate distillation LR (no-op for non-L-SSM backends).
    #[cfg(feature = "liquid-mamba")]
    fn set_fep_modulation(&self, _fep_signal: f32) {}

    /// Cycle-level distillation modulation: adjusts the FEP factor that the
    /// next per-generation `distill_step()` uses. Gated by prediction confidence
    /// and thermodynamic load. This is a *modulation* step, not a gradient step.
    ///
    /// - `fep_precision`: free-energy proxy (0-1, higher = more surprise)
    /// - `thermodynamic_load`: current metabolic cost (0-1)
    /// - `prediction_confidence`: consciousness-level confidence gate
    /// - `fep_lr_boost`: scaling factor for the modulation (default 1.0)
    #[cfg(feature = "liquid-mamba")]
    fn cycle_level_distill(
        &self,
        _fep_precision: f32,
        _thermodynamic_load: f32,
        _prediction_confidence: f32,
        _fep_lr_boost: f32,
    ) {
    }

    /// Current effective distillation learning rate (0.0 for non-L-SSM backends).
    #[cfg(feature = "liquid-mamba")]
    fn current_distillation_lr(&self) -> f32 {
        0.0
    }

    /// Last cached effective rank of projection bottleneck activations (0.0 for non-L-SSM).
    #[cfg(feature = "liquid-mamba")]
    fn last_effective_rank(&self) -> f32 {
        0.0
    }

    /// Total number of generation/distillation cycles completed (0 for non-L-SSM).
    #[cfg(feature = "liquid-mamba")]
    fn generation_count(&self) -> u32 {
        0
    }

    /// Get the backend name for logging.
    fn name(&self) -> &str;
}

/// Ollama backend connecting to a local Ollama instance.
///
/// Sends requests to `http://localhost:11434/api/generate` with
/// configurable model, timeout, and generation parameters.
pub struct OllamaBackend {
    /// Base URL for the Ollama API.
    base_url: String,
    /// Model to use for generation.
    model: String,
    /// HTTP client with timeout.
    client: reqwest::Client,
}

/// Ollama /api/generate request body (legacy, used for streaming).
#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    stream: bool,
    options: OllamaOptions,
}

/// Ollama /api/chat request body (preferred — works with Gemma 4).
#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaChatMessage<'a>>,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// Ollama generation options.
#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: usize,
}

/// Ollama /api/generate response body.
#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
    #[allow(dead_code)]
    done: bool,
}

/// Ollama /api/chat response body.
#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatResponseMessage,
}

#[derive(Deserialize)]
struct OllamaChatResponseMessage {
    content: String,
}

impl OllamaBackend {
    /// Create a new Ollama backend with default settings.
    ///
    /// Connects to `http://localhost:11434` with a model determined by the
    /// `SYMTHAEA_LLM_MODEL` environment variable, falling back to `gemma4:e2b`.
    pub fn new() -> Self {
        let model =
            std::env::var("SYMTHAEA_LLM_MODEL").unwrap_or_else(|_| "gemma4:e2b".to_string());
        Self::with_config("http://localhost:11434", &model)
    }

    /// Create with a specific model name, using the default Ollama URL.
    pub fn with_model(model: &str) -> Self {
        Self::with_config("http://localhost:11434", model)
    }

    /// Create with custom URL and model.
    pub fn with_config(base_url: &str, model: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(180)) // 3 minutes for longer prompts
            .build()
            .unwrap_or_else(|e| {
                tracing::warn!(error = %e, "Failed to build HTTP client with custom config, falling back to default");
                reqwest::Client::default()
            });

        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            client,
        }
    }

    /// Create with custom URL, model, and connect timeout.
    ///
    /// The connect timeout controls how long to wait for the initial TCP connection.
    /// The response timeout controls the total request duration.
    pub fn with_timeouts(
        base_url: &str,
        model: &str,
        connect_timeout: std::time::Duration,
        response_timeout: std::time::Duration,
    ) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(connect_timeout)
            .timeout(response_timeout)
            .build()
            .unwrap_or_else(|e| {
                tracing::warn!(error = %e, "Failed to build HTTP client with timeouts, falling back to default");
                reqwest::Client::default()
            });

        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            client,
        }
    }
}

#[async_trait::async_trait]
impl LLMBackend for OllamaBackend {
    async fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<String> {
        // Use /api/chat (not /api/generate) — Gemma 4 returns empty with generate API.
        let url = format!("{}/api/chat", self.base_url);

        // Merge consciousness context into system prompt if present
        let effective_system: Option<String> =
            match (&params.system_prompt, &params.consciousness_context) {
                (Some(sys), Some(ctx)) => Some(format!(
                    "{}\n\n# Consciousness Context\n{}",
                    sys,
                    ctx.to_system_supplement()
                )),
                (None, Some(ctx)) => Some(format!(
                    "# Consciousness Context\n{}",
                    ctx.to_system_supplement()
                )),
                (Some(sys), None) => Some(sys.clone()),
                (None, None) => None,
            };

        // Adjust temperature based on consciousness context
        let effective_temp = if let Some(ref ctx) = params.consciousness_context {
            (params.temperature + ctx.temperature_adjustment()).clamp(0.1, 1.5)
        } else {
            params.temperature
        };

        // Build messages array (system + user)
        let mut messages = Vec::with_capacity(2);
        let system_text;
        if let Some(ref sys) = effective_system {
            system_text = sys.clone();
            messages.push(OllamaChatMessage {
                role: "system",
                content: &system_text,
            });
        }
        messages.push(OllamaChatMessage {
            role: "user",
            content: prompt,
        });

        let request_body = OllamaChatRequest {
            model: &self.model,
            messages,
            stream: false,
            options: OllamaOptions {
                temperature: effective_temp,
                num_predict: params.max_tokens,
            },
        };

        tracing::info!(
            target: "llm_backend",
            model = %self.model,
            num_predict = params.max_tokens,
            "Ollama chat request"
        );

        let response = self
            .client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Ollama chat request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<body unreadable>".to_string());
            anyhow::bail!("Ollama returned {status}: {body}");
        }

        let chat_response: OllamaChatResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse Ollama chat response: {e}"))?;

        tracing::info!(
            target: "llm_backend",
            len = chat_response.message.content.len(),
            "Ollama chat response"
        );

        Ok(chat_response.message.content)
    }

    async fn generate_streaming(
        &self,
        prompt: &str,
        params: &GenerationParams,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<String> {
        let url = format!("{}/api/generate", self.base_url);

        let request_body = OllamaRequest {
            model: &self.model,
            prompt,
            system: params.system_prompt.as_deref(),
            stream: true,
            options: OllamaOptions {
                temperature: params.temperature,
                num_predict: params.max_tokens,
            },
        };

        let response = self
            .client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Ollama streaming request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<body unreadable>".to_string());
            anyhow::bail!("Ollama returned {status}: {body}");
        }

        // Read chunks as they arrive from the network and process line by line.
        // Ollama sends one JSON object per line: {"response":"token","done":false}\n
        let mut full_response = String::new();
        let mut buffer = Vec::new();

        let mut stream = response;
        while let Some(chunk) = stream
            .chunk()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read Ollama stream chunk: {e}"))?
        {
            buffer.extend_from_slice(&chunk);

            // Process complete lines from the buffer
            while let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buffer.drain(..=newline_pos).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(parsed) = serde_json::from_str::<OllamaResponse>(line) {
                    let token = parsed.response;
                    if !token.is_empty() {
                        full_response.push_str(&token);
                        on_token(&token);
                    }
                }
            }
        }

        // Process any remaining data in buffer (last line may lack trailing newline)
        if !buffer.is_empty() {
            let line = String::from_utf8_lossy(&buffer);
            let line = line.trim();
            if !line.is_empty() {
                if let Ok(parsed) = serde_json::from_str::<OllamaResponse>(line) {
                    let token = parsed.response;
                    if !token.is_empty() {
                        full_response.push_str(&token);
                        on_token(&token);
                    }
                }
            }
        }

        Ok(full_response)
    }

    async fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    fn name(&self) -> &str {
        "Ollama"
    }
}

/// Simulated backend for when no LLM is available.
///
/// Produces deterministic responses based on prompt content.
/// Used as a fallback when Ollama is not running.
pub struct SimulatedBackend;

#[async_trait::async_trait]
impl LLMBackend for SimulatedBackend {
    async fn generate(&self, prompt: &str, _params: &GenerationParams) -> Result<String> {
        let lower = prompt.to_lowercase();
        // Produce a simple response based on the prompt
        let response =
            if lower.contains("fix") && (lower.contains("flake") || lower.contains("nix")) {
                // Return a fixed Nix flake for the demo
                r#"{
  description = "A fixed flake for Symthaea";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.mkShell {
      buildInputs = [ 
        nixpkgs.legacyPackages.x86_64-linux.hello
      ];
      shellHook = "hello"; 
    };
  };
}"#
                .to_string()
            } else if prompt.contains("translate") || prompt.contains("Translate") {
                format!("I understand your input. {}", summarize_prompt(prompt))
            } else if prompt.contains('?') {
                format!("Regarding your question: {}", summarize_prompt(prompt))
            } else {
                format!("Acknowledged. {}", summarize_prompt(prompt))
            };
        Ok(response)
    }

    async fn is_available(&self) -> bool {
        true // Always available
    }

    fn name(&self) -> &str {
        "Simulated"
    }
}

use std::sync::Arc;

/// Create the default backend: tries Ollama first, falls back to simulated.
pub fn default_backend() -> Arc<dyn LLMBackend> {
    Arc::new(OllamaBackend::new())
}

/// Create a Liquid-Mamba backend with a specific configuration.
///
/// Returns an `Arc<dyn LLMBackend>` that can be passed to `LLMOrgan::with_backend`.
/// Errors if the Mamba model fails to load (network required on first run).
#[cfg(feature = "liquid-mamba")]
pub fn backend_with_liquid_mamba_config(
    genesis: &symthaea_core::genesis::GenesisSeed,
    config: symthaea_broca::LiquidMambaConfig,
) -> anyhow::Result<Arc<dyn LLMBackend>> {
    let backend = super::ssm_backend::LiquidMambaBackend::new(genesis, config)?;
    Ok(Arc::new(backend))
}

/// Create a simulated-only backend (for testing or offline use).
pub fn simulated_backend() -> Arc<dyn LLMBackend> {
    Arc::new(SimulatedBackend)
}

/// Create a backend based on environment configuration.
///
/// Priority:
/// 1. `SYMTHAEA_LLM_PROVIDER` env var (`ollama`, `openai`, `anthropic`)
/// 2. OpenAI (if `OPENAI_API_KEY` is set)
/// 3. Anthropic (if `ANTHROPIC_API_KEY` is set)
/// 4. Ollama (default, works offline with local models)
pub fn create_backend_from_env() -> Arc<dyn LLMBackend> {
    if let Ok(provider) = std::env::var("SYMTHAEA_LLM_PROVIDER") {
        match provider.to_lowercase().as_str() {
            "openai" => {
                if let Some(backend) = super::openai_backend::OpenAiBackend::from_env() {
                    return Arc::new(backend);
                }
                tracing::warn!("SYMTHAEA_LLM_PROVIDER=openai but OPENAI_API_KEY not set, falling back to Ollama");
            }
            "anthropic" => {
                if let Some(backend) = super::anthropic_backend::AnthropicBackend::from_env() {
                    return Arc::new(backend);
                }
                tracing::warn!("SYMTHAEA_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY not set, falling back to Ollama");
            }
            "ollama" => {
                let model =
                    std::env::var("SYMTHAEA_LLM_MODEL").unwrap_or_else(|_| "llama3".to_string());
                return Arc::new(OllamaBackend::with_model(&model));
            }
            "candle" => {
                #[cfg(feature = "full_language")]
                {
                    let model_id = std::env::var("SYMTHAEA_CANDLE_MODEL")
                        .unwrap_or_else(|_| "bartowski/Llama-3.2-1B-Instruct-GGUF".to_string());
                    let file_name = std::env::var("SYMTHAEA_CANDLE_FILE")
                        .unwrap_or_else(|_| "Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string());
                    if let Ok(backend) =
                        super::candle_backend::CandleBackend::new(&model_id, "main", &file_name)
                    {
                        return Arc::new(backend);
                    }
                }
                tracing::warn!("SYMTHAEA_LLM_PROVIDER=candle but feature full_language not enabled or load failed, falling back to Ollama");
                return Arc::new(OllamaBackend::new());
            }
            "ssm" | "broca" => {
                #[cfg(feature = "ssm_language")]
                {
                    let genesis =
                        symthaea_core::genesis::GenesisSeed::from_phrase("symthaea-broca-default");
                    return Arc::new(super::ssm_backend::SsmBackend::new(&genesis));
                }
                #[cfg(not(feature = "ssm_language"))]
                {
                    tracing::warn!("SYMTHAEA_LLM_PROVIDER=ssm but feature ssm_language not enabled, falling back to Ollama");
                }
            }
            "liquid-mamba" | "l-ssm" => {
                #[cfg(feature = "liquid-mamba")]
                {
                    let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(
                        "symthaea-broca-liquid-mamba",
                    );
                    let config = symthaea_broca::LiquidMambaConfig::default();
                    match super::ssm_backend::LiquidMambaBackend::new(&genesis, config) {
                        Ok(backend) => return Arc::new(backend),
                        Err(e) => {
                            tracing::error!("Failed to load Liquid-Mamba backend: {e}");
                        }
                    }
                }
                #[cfg(not(feature = "liquid-mamba"))]
                {
                    tracing::warn!("SYMTHAEA_LLM_PROVIDER=liquid-mamba but feature liquid-mamba not enabled, falling back to Ollama");
                }
            }
            "simulated" => {
                return Arc::new(SimulatedBackend);
            }
            other => {
                tracing::warn!("Unknown LLM provider '{}', falling back to Ollama", other);
            }
        }
    }

    // Auto-detect: try API keys first, then Ollama
    if let Some(backend) = super::openai_backend::OpenAiBackend::from_env() {
        return Arc::new(backend);
    }
    if let Some(backend) = super::anthropic_backend::AnthropicBackend::from_env() {
        return Arc::new(backend);
    }

    let model = std::env::var("SYMTHAEA_LLM_MODEL").unwrap_or_else(|_| "gemma4:e2b".to_string());
    Arc::new(OllamaBackend::with_model(&model))
}

/// Summarize a prompt to ~20 words for simulated responses.
fn summarize_prompt(prompt: &str) -> String {
    let words: Vec<&str> = prompt.split_whitespace().collect();
    if words.len() <= 20 {
        words.join(" ")
    } else {
        format!("{}...", words[..20].join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // GenerationParams Tests
    // =========================================================================

    #[test]
    fn test_generation_params_default() {
        let params = GenerationParams::default();
        assert!((params.temperature - 0.7).abs() < 0.01);
        assert_eq!(params.max_tokens, 256);
        assert!(params.system_prompt.is_none());
    }

    #[test]
    fn test_generation_params_custom() {
        let params = GenerationParams {
            temperature: 0.3,
            max_tokens: 100,
            system_prompt: Some("You are helpful".to_string()),
            consciousness_context: None,
        };
        assert!((params.temperature - 0.3).abs() < 0.01);
        assert_eq!(params.max_tokens, 100);
        assert_eq!(params.system_prompt, Some("You are helpful".to_string()));
    }

    #[test]
    fn test_generation_params_clone() {
        let params = GenerationParams {
            temperature: 0.5,
            max_tokens: 200,
            system_prompt: Some("Test".to_string()),
            consciousness_context: None,
        };
        let cloned = params.clone();
        assert!((params.temperature - cloned.temperature).abs() < 0.01);
        assert_eq!(params.max_tokens, cloned.max_tokens);
    }

    // =========================================================================
    // ConsciousnessContext Tests
    // =========================================================================

    #[test]
    fn test_consciousness_context_system_supplement() {
        let ctx = ConsciousnessContext {
            epistemic_status: "Certain".to_string(),
            phi: 0.7,
            type_confidence: 0.9,
            algorithm_pattern: 0.5,
            error_likelihood: 0.1,
            syntax_complexity: 0.3,
        };
        let supplement = ctx.to_system_supplement();
        assert!(supplement.contains("EPISTEMIC_STATUS: Certain"));
        assert!(supplement.contains("CONSCIOUSNESS_LEVEL: 0.70"));
        assert!(supplement.contains("TYPE_CONFIDENCE: 0.90"));
        // Low error likelihood → no error note
        assert!(!supplement.contains("High error likelihood"));
    }

    #[test]
    fn test_consciousness_context_high_error_note() {
        let ctx = ConsciousnessContext {
            epistemic_status: "Uncertain".to_string(),
            phi: 0.3,
            type_confidence: 0.2,
            algorithm_pattern: 0.0,
            error_likelihood: 0.8,
            syntax_complexity: 0.1,
        };
        let supplement = ctx.to_system_supplement();
        assert!(supplement.contains("High error likelihood"));
        assert!(supplement.contains("Low type confidence"));
        assert!(supplement.contains("Low epistemic confidence"));
    }

    #[test]
    fn test_consciousness_context_temperature_adjustment() {
        // High confidence → cooler (more deterministic)
        let ctx = ConsciousnessContext {
            epistemic_status: "Certain".to_string(),
            phi: 0.8,
            type_confidence: 1.0,
            algorithm_pattern: 0.5,
            error_likelihood: 0.0,
            syntax_complexity: 0.3,
        };
        let adj = ctx.temperature_adjustment();
        assert!(adj < 0.0, "High type_confidence should lower temperature");
        assert!(adj > -0.5, "Adjustment should be moderate");

        // Low confidence → barely any adjustment
        let ctx2 = ConsciousnessContext {
            epistemic_status: "Unknown".to_string(),
            phi: 0.2,
            type_confidence: 0.0,
            algorithm_pattern: 0.0,
            error_likelihood: 0.0,
            syntax_complexity: 0.0,
        };
        let adj2 = ctx2.temperature_adjustment();
        assert!((adj2 - 0.0).abs() < 0.01, "No code signals → no adjustment");
    }

    // =========================================================================
    // SimulatedBackend Tests
    // =========================================================================

    #[tokio::test]
    async fn test_simulated_backend_always_available() {
        let backend = SimulatedBackend;
        assert!(backend.is_available().await);
    }

    #[tokio::test]
    async fn test_simulated_backend_generates_response() {
        let backend = SimulatedBackend;
        let params = GenerationParams::default();
        let result = backend.generate("Hello, how are you?", &params).await;
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(!text.is_empty());
    }

    #[tokio::test]
    async fn test_simulated_backend_name() {
        let backend = SimulatedBackend;
        assert_eq!(backend.name(), "Simulated");
    }

    #[tokio::test]
    async fn test_simulated_backend_translate_prompt() {
        let backend = SimulatedBackend;
        let params = GenerationParams::default();
        let result = backend
            .generate("Please translate this text", &params)
            .await;
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("understand") || text.contains("input"));
    }

    #[tokio::test]
    async fn test_simulated_backend_question_prompt() {
        let backend = SimulatedBackend;
        let params = GenerationParams::default();
        let result = backend
            .generate("What is the meaning of life?", &params)
            .await;
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("question") || text.contains("Regarding"));
    }

    #[tokio::test]
    async fn test_simulated_backend_statement_prompt() {
        let backend = SimulatedBackend;
        let params = GenerationParams::default();
        let result = backend.generate("The sky is blue.", &params).await;
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("Acknowledged"));
    }

    // =========================================================================
    // OllamaBackend Tests
    // =========================================================================

    #[tokio::test]
    async fn test_ollama_backend_creation() {
        let backend = OllamaBackend::new();
        assert_eq!(backend.name(), "Ollama");
    }

    #[test]
    fn test_ollama_backend_with_config() {
        let backend = OllamaBackend::with_config("http://localhost:8080", "llama2");
        assert_eq!(backend.name(), "Ollama");
        assert_eq!(backend.base_url, "http://localhost:8080");
        assert_eq!(backend.model, "llama2");
    }

    #[test]
    fn test_ollama_backend_default_config() {
        let backend = OllamaBackend::new();
        assert_eq!(backend.base_url, "http://localhost:11434");
        // Model comes from SYMTHAEA_LLM_MODEL env var or defaults to "qwen2.5-coder:7b".
        // Don't assert exact value — parallel tests may set the env var.
        assert!(!backend.model.is_empty());
    }

    #[test]
    fn test_ollama_backend_env_var_model_override() {
        let prev = std::env::var("SYMTHAEA_LLM_MODEL").ok();
        std::env::set_var("SYMTHAEA_LLM_MODEL", "mistral:7b");

        let backend = OllamaBackend::new();
        assert_eq!(backend.model, "mistral:7b");
        assert_eq!(backend.base_url, "http://localhost:11434");

        // Restore
        match prev {
            Some(val) => std::env::set_var("SYMTHAEA_LLM_MODEL", val),
            None => std::env::remove_var("SYMTHAEA_LLM_MODEL"),
        }
    }

    #[test]
    fn test_ollama_backend_with_model() {
        let backend = OllamaBackend::with_model("qwen3:1.7b");
        assert_eq!(backend.model, "qwen3:1.7b");
        assert_eq!(backend.base_url, "http://localhost:11434");
    }

    /// Integration test for Ollama backend.
    /// Requires Ollama to be running with llama3.2:3b or similar model.
    /// Skips gracefully if Ollama is not available.
    #[tokio::test]
    async fn test_ollama_integration() {
        let backend = OllamaBackend::new();

        // Skip if Ollama is not available
        if !backend.is_available().await {
            eprintln!("Skipping Ollama integration test: Ollama not available");
            return;
        }

        let params = GenerationParams {
            temperature: 0.3,
            max_tokens: 50,
            system_prompt: Some(
                "You are a helpful assistant. Keep responses very brief.".to_string(),
            ),
            consciousness_context: None,
        };

        let result = backend
            .generate("Say hello in exactly 3 words.", &params)
            .await;

        match result {
            Ok(response) => {
                assert!(!response.is_empty(), "Ollama returned empty response");
                println!("Ollama response: {}", response);
            }
            Err(e) => {
                // Model might not be available - this is acceptable in CI
                eprintln!(
                    "Ollama generation failed (model may not be installed): {}",
                    e
                );
            }
        }
    }

    // =========================================================================
    // Factory Function Tests
    // =========================================================================

    #[test]
    fn test_default_backend() {
        let backend = default_backend();
        assert_eq!(backend.name(), "Ollama");
    }

    #[test]
    fn test_simulated_backend_factory() {
        let backend = simulated_backend();
        assert_eq!(backend.name(), "Simulated");
    }

    /// Mutex to serialize tests that mutate environment variables.
    /// `std::env::set_var` / `remove_var` is process-global and not
    /// thread-safe, so parallel test threads can race.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_create_backend_from_env_default() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev_provider = std::env::var("SYMTHAEA_LLM_PROVIDER").ok();
        let prev_openai = std::env::var("OPENAI_API_KEY").ok();
        let prev_anthropic = std::env::var("ANTHROPIC_API_KEY").ok();

        std::env::remove_var("SYMTHAEA_LLM_PROVIDER");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("ANTHROPIC_API_KEY");

        let backend = create_backend_from_env();
        assert_eq!(backend.name(), "Ollama");

        if let Some(v) = prev_provider {
            std::env::set_var("SYMTHAEA_LLM_PROVIDER", v);
        }
        if let Some(v) = prev_openai {
            std::env::set_var("OPENAI_API_KEY", v);
        }
        if let Some(v) = prev_anthropic {
            std::env::set_var("ANTHROPIC_API_KEY", v);
        }
    }

    #[test]
    fn test_create_backend_from_env_simulated() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev = std::env::var("SYMTHAEA_LLM_PROVIDER").ok();
        std::env::set_var("SYMTHAEA_LLM_PROVIDER", "simulated");

        let backend = create_backend_from_env();
        assert_eq!(backend.name(), "Simulated");

        match prev {
            Some(v) => std::env::set_var("SYMTHAEA_LLM_PROVIDER", v),
            None => std::env::remove_var("SYMTHAEA_LLM_PROVIDER"),
        }
    }

    // =========================================================================
    // summarize_prompt Tests
    // =========================================================================

    #[test]
    fn test_summarize_prompt_short() {
        let result = summarize_prompt("Hello world");
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_summarize_prompt_long() {
        let long_text = (0..30)
            .map(|i| format!("word{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let result = summarize_prompt(&long_text);
        assert!(result.ends_with("..."));
        // Should be truncated to ~20 words
        let word_count = result.trim_end_matches("...").split_whitespace().count();
        assert_eq!(word_count, 20);
    }

    #[test]
    fn test_summarize_prompt_exactly_20_words() {
        let text = (0..20)
            .map(|i| format!("word{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let result = summarize_prompt(&text);
        assert!(!result.ends_with("..."));
        assert_eq!(result, text);
    }

    #[test]
    fn test_summarize_prompt_empty() {
        let result = summarize_prompt("");
        assert!(result.is_empty());
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[tokio::test]
    async fn test_simulated_backend_empty_prompt() {
        let backend = SimulatedBackend;
        let params = GenerationParams::default();
        let result = backend.generate("", &params).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_simulated_backend_special_characters() {
        let backend = SimulatedBackend;
        let params = GenerationParams::default();
        let result = backend.generate("Hello! @#$%^&*() 你好 🎉", &params).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_simulated_backend_very_long_prompt() {
        let backend = SimulatedBackend;
        let params = GenerationParams::default();
        let long_prompt = "word ".repeat(1000);
        let result = backend.generate(&long_prompt, &params).await;
        assert!(result.is_ok());
        // Response should be truncated
        let text = result.unwrap();
        assert!(text.contains("..."));
    }

    #[tokio::test]
    async fn test_simulated_backend_ignores_params() {
        let backend = SimulatedBackend;

        // Test with various params - should all produce similar output style
        let params_low_temp = GenerationParams {
            temperature: 0.0,
            max_tokens: 10,
            system_prompt: Some("Be concise".to_string()),
            consciousness_context: None,
        };

        let params_high_temp = GenerationParams {
            temperature: 1.0,
            max_tokens: 1000,
            system_prompt: None,
            consciousness_context: None,
        };

        let result1 = backend.generate("Hello", &params_low_temp).await.unwrap();
        let result2 = backend.generate("Hello", &params_high_temp).await.unwrap();

        // Both should produce valid responses (simulated ignores params)
        assert!(!result1.is_empty());
        assert!(!result2.is_empty());
    }
}
