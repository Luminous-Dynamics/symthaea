// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ollama LLM bridge for general-purpose queries.
//!
//! When HDC-based intent recognition and causal reasoning can't handle a query
//! (e.g., open-ended questions like "what's the difference between ext4 and btrfs?"),
//! this module sends it to a local Ollama instance for LLM-powered response.
//!
//! Approved models (from CLAUDE.md):
//!   embeddinggemma:300m | gemma3:1b | qwen3:1.7b | gemma4:e2b | mistral:7b
//!
//! Do NOT use: qwen2.5 variants

use std::time::Duration;

/// Configuration for the Ollama bridge.
#[derive(Debug, Clone)]
pub struct OllamaBridgeConfig {
    /// Ollama API endpoint (default: http://localhost:11434)
    pub endpoint: String,
    /// Primary model to use
    pub model: String,
    /// Fallback models in priority order
    pub fallback_models: Vec<String>,
    /// Request timeout
    pub timeout: Duration,
    /// System prompt prefix for NixOS context
    pub system_prompt: String,
}

impl Default for OllamaBridgeConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:11434".into(),
            model: "gemma3:1b".into(),
            fallback_models: vec![
                "gemma3:1b".into(),
                "qwen3:1.7b".into(),
                "gemma4:e2b".into(),
                "mistral:7b".into(),
            ],
            timeout: Duration::from_secs(30),
            system_prompt: concat!(
                "You are a NixOS expert assistant. Answer questions about NixOS ",
                "configuration, packages, services, flakes, and system management. ",
                "Be concise and provide practical examples when relevant. ",
                "Use NixOS 25.11 conventions."
            )
            .into(),
        }
    }
}

/// Response from Ollama.
#[derive(Debug, Clone)]
pub struct OllamaResponse {
    pub text: String,
    pub model_used: String,
    pub duration_ms: u64,
}

/// Bridge to local Ollama instance for LLM-powered queries.
pub struct OllamaBridge {
    config: OllamaBridgeConfig,
    available: Option<bool>,
    /// Cached set of locally available model base names (lowercase).
    cached_models: Option<Vec<String>>,
}

impl OllamaBridge {
    pub fn new(config: OllamaBridgeConfig) -> Self {
        Self {
            config,
            available: None,
            cached_models: None,
        }
    }

    /// Invalidate cached availability and model list.
    pub fn invalidate_cache(&mut self) {
        self.available = None;
        self.cached_models = None;
    }

    /// Refresh the cached model list from `ollama list`.
    fn refresh_model_cache(&mut self) -> Option<&[String]> {
        let output = std::process::Command::new("ollama")
            .arg("list")
            .output()
            .ok()?;

        if !output.status.success() {
            self.cached_models = Some(Vec::new());
            return self.cached_models.as_deref();
        }

        let stdout = String::from_utf8_lossy(&output.stdout).to_lowercase();
        let models: Vec<String> = self
            .config
            .fallback_models
            .iter()
            .filter(|m| {
                let base = m.split(':').next().unwrap_or(m);
                stdout.contains(base)
            })
            .cloned()
            .collect();

        self.cached_models = Some(models);
        self.cached_models.as_deref()
    }

    /// Get available models (from cache or by querying ollama).
    fn available_models(&mut self) -> &[String] {
        if self.cached_models.is_none() {
            self.refresh_model_cache();
        }
        self.cached_models.as_deref().unwrap_or(&[])
    }

    /// Check if Ollama is reachable and has a usable model.
    pub fn check_available(&mut self) -> bool {
        if let Some(cached) = self.available {
            return cached;
        }

        let available = !self.available_models().is_empty();
        self.available = Some(available);
        available
    }

    /// Try a single query against a specific model. Returns `None` on failure.
    fn try_query_model(&self, model: &str, prompt: &str) -> Option<OllamaResponse> {
        let start = std::time::Instant::now();

        let payload = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "system": self.config.system_prompt,
            "stream": false,
            "options": {
                "temperature": 0.3,
                "num_predict": 512
            }
        });

        let url = format!("{}/api/generate", self.config.endpoint);

        let child = std::process::Command::new("curl")
            .args([
                "-s",
                "--max-time",
                &self.config.timeout.as_secs().to_string(),
                "-X",
                "POST",
                &url,
                "-H",
                "Content-Type: application/json",
                "-d",
                &payload.to_string(),
            ])
            .output();

        let output = child.ok()?;
        if !output.status.success() {
            tracing::warn!(
                model,
                "Ollama request failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return None;
        }

        let body: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
        let text = body.get("response")?.as_str()?.to_string();

        if text.is_empty() {
            return None;
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        Some(OllamaResponse {
            text,
            model_used: model.to_string(),
            duration_ms,
        })
    }

    /// Send a query to Ollama, trying each available model in fallback order.
    ///
    /// Returns `None` if Ollama is unavailable or all models fail.
    pub fn query(&mut self, prompt: &str) -> Option<OllamaResponse> {
        if !self.check_available() {
            return None;
        }

        // Clone the model list to avoid borrow conflict
        let models = self.available_models().to_vec();

        for model in &models {
            if let Some(response) = self.try_query_model(model, prompt) {
                return Some(response);
            }
            tracing::info!(model = model.as_str(), "Model failed, trying next fallback");
        }

        // All models failed — invalidate cache so next call re-discovers
        self.invalidate_cache();
        None
    }

    /// Query with NixOS-specific context prepended.
    pub fn query_nixos(&mut self, question: &str, context: Option<&str>) -> Option<OllamaResponse> {
        let prompt = if let Some(ctx) = context {
            format!(
                "Context about the user's NixOS system:\n{}\n\nQuestion: {}",
                ctx, question
            )
        } else {
            question.to_string()
        };
        self.query(&prompt)
    }
}

impl Default for OllamaBridge {
    fn default() -> Self {
        Self::new(OllamaBridgeConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = OllamaBridgeConfig::default();
        assert_eq!(config.endpoint, "http://localhost:11434");
        assert_eq!(config.model, "gemma3:1b");
        assert!(!config.fallback_models.is_empty());
        assert!(config.system_prompt.contains("NixOS"));
    }

    #[test]
    fn test_bridge_creation() {
        let bridge = OllamaBridge::default();
        assert!(bridge.available.is_none());
        assert!(bridge.cached_models.is_none());
    }

    #[test]
    fn test_no_qwen25_in_defaults() {
        let config = OllamaBridgeConfig::default();
        for model in &config.fallback_models {
            assert!(
                !model.contains("qwen2.5"),
                "qwen2.5 variants must not be used per CLAUDE.md"
            );
        }
    }

    #[test]
    fn test_invalidate_cache() {
        let mut bridge = OllamaBridge::default();
        bridge.available = Some(true);
        bridge.cached_models = Some(vec!["gemma3:1b".into()]);

        bridge.invalidate_cache();

        assert!(bridge.available.is_none());
        assert!(bridge.cached_models.is_none());
    }

    #[test]
    fn test_try_query_model_unreachable() {
        // Use a bogus endpoint so curl fails fast
        let config = OllamaBridgeConfig {
            endpoint: "http://127.0.0.1:1".into(),
            timeout: Duration::from_secs(1),
            ..Default::default()
        };
        let bridge = OllamaBridge::new(config);
        let result = bridge.try_query_model("gemma3:1b", "test");
        assert!(result.is_none());
    }

    #[test]
    fn test_fallback_model_order() {
        let config = OllamaBridgeConfig::default();
        assert_eq!(config.fallback_models[0], "gemma3:1b");
        assert_eq!(config.fallback_models[1], "qwen3:1.7b");
        assert_eq!(config.fallback_models[2], "gemma4:e2b");
        assert_eq!(config.fallback_models[3], "mistral:7b");
    }
}
