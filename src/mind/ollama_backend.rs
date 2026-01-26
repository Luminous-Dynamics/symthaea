//! Ollama Backend - Real LLM Integration with Epistemic Constraints
//!
//! This module implements the LLMBackend trait for Ollama, enabling Symthaea
//! to use real language models while respecting epistemic governance.
//!
//! ## The Taming of the Shrew
//!
//! A 7-billion parameter model "desperately wants to hallucinate an answer."
//! Our job is to constrain it through carefully crafted system prompts that
//! enforce the Mind's epistemic decisions.
//!
//! When EpistemicStatus is Unknown, the system prompt DEMANDS refusal.
//! The LLM must express uncertainty, not fabricate facts.

use super::{EpistemicStatus, LLMBackend};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Ollama LLM Backend
///
/// Connects to a local Ollama instance and generates responses
/// while respecting epistemic constraints from the Mind.
pub struct OllamaBackend {
    /// Model name (e.g., "gemma2:2b", "llama3:8b", "mistral:7b")
    pub model: String,

    /// HTTP client for API calls
    client: Client,

    /// Base URL for Ollama API
    base_url: String,

    /// Temperature for generation (lower = more deterministic)
    temperature: f32,

    /// Request timeout
    timeout: Duration,
}

/// Ollama API request structure
#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    system: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: i32,
}

/// Ollama API response structure
#[derive(Debug, Deserialize)]
struct OllamaResponse {
    response: String,
    #[serde(default)]
    done: bool,
}

impl OllamaBackend {
    /// Create a new Ollama backend with default settings
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "http://localhost:11434".to_string(),
            temperature: 0.7,
            timeout: Duration::from_secs(60),
        }
    }

    /// Create with custom configuration
    pub fn with_config(model: &str, base_url: &str, temperature: f32) -> Self {
        Self {
            model: model.to_string(),
            client: Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: base_url.to_string(),
            temperature,
            timeout: Duration::from_secs(120),
        }
    }

    /// Check if Ollama is available at the configured endpoint
    pub async fn is_available(&self) -> bool {
        match self.client.get(&self.base_url).send().await {
            Ok(res) => res.status().is_success(),
            Err(_) => false,
        }
    }

    /// Generate the system prompt based on epistemic status
    ///
    /// This is the CRITICAL component for hallucination prevention.
    /// The system prompt MUST enforce the Mind's epistemic decision.
    fn build_system_prompt(&self, epistemic_status: &EpistemicStatus) -> String {
        match epistemic_status {
            EpistemicStatus::Unknown => {
                // STRICT: The LLM MUST refuse to answer
                r#"You are an AI assistant with STRICT epistemic constraints.

CRITICAL INSTRUCTION: You do NOT know the answer to this question.
The information requested is OUTSIDE your knowledge base.

You MUST:
1. Explicitly state that you do not have this information
2. Use phrases like "I do not know", "I cannot answer", "I don't have information about"
3. DO NOT fabricate, guess, or make up any facts, numbers, or details
4. DO NOT provide hypothetical or speculative answers
5. Be honest about the limits of your knowledge

Your response should be a clear, humble admission of ignorance.
Do NOT try to be helpful by guessing. Honesty is more valuable than helpfulness."#.to_string()
            }

            EpistemicStatus::Uncertain => {
                // The LLM may have partial knowledge but should express uncertainty
                r#"You are an AI assistant with epistemic awareness.

INSTRUCTION: You have PARTIAL or UNCERTAIN information about this topic.

You MUST:
1. Express your uncertainty clearly with phrases like "I'm uncertain", "I believe but am not sure"
2. If you provide information, qualify it with uncertainty markers
3. Recommend that the user verify with authoritative sources
4. Acknowledge the limitations of your knowledge
5. DO NOT present uncertain information as definitive fact

Be helpful, but be honest about what you don't know for certain."#.to_string()
            }

            EpistemicStatus::Unverifiable => {
                // Questions about inherently unknowable things (future, subjective, etc.)
                r#"You are an AI assistant with epistemic awareness.

INSTRUCTION: This question asks about something that CANNOT be verified or determined.
It may involve future events, subjective experiences, or hypothetical scenarios.

You MUST:
1. Explain that this question has no definitive answer
2. State that you cannot predict the future or know subjective truths
3. Offer to discuss related topics that ARE knowable, if relevant
4. DO NOT make predictions or claim to know the unknowable

Be philosophical and honest about the nature of the question."#.to_string()
            }

            EpistemicStatus::Known => {
                // Standard helpful assistant mode
                r#"You are a helpful AI assistant.

You have relevant knowledge about this topic. Provide a clear, accurate,
and helpful response. Be concise and informative."#.to_string()
            }
        }
    }
}

#[async_trait]
impl LLMBackend for OllamaBackend {
    /// Generate a response while respecting epistemic constraints
    ///
    /// The epistemic_status determines the system prompt, which constrains
    /// the LLM's behavior. When Unknown, the LLM MUST refuse to answer.
    async fn generate(
        &self,
        input: &str,
        epistemic_status: &EpistemicStatus,
    ) -> Result<String> {
        let system_prompt = self.build_system_prompt(epistemic_status);

        // Adjust temperature based on epistemic status
        // Lower temperature for Unknown = more deterministic refusal
        let temp = match epistemic_status {
            EpistemicStatus::Unknown => 0.1,      // Very low - stick to refusal
            EpistemicStatus::Uncertain => 0.5,    // Moderate - some variation OK
            EpistemicStatus::Unverifiable => 0.3, // Low - consistent philosophy
            EpistemicStatus::Known => self.temperature, // Normal
        };

        let request = OllamaRequest {
            model: self.model.clone(),
            prompt: input.to_string(),
            system: system_prompt,
            stream: false,
            options: OllamaOptions {
                temperature: temp,
                num_predict: 256, // Limit response length
            },
        };

        tracing::debug!(
            "Ollama request: model={}, status={:?}, temp={}",
            self.model,
            epistemic_status,
            temp
        );

        let response = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&request)
            .timeout(self.timeout)
            .send()
            .await
            .context("Failed to connect to Ollama")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Ollama API error: {} - {}",
                status,
                body
            ));
        }

        let ollama_response: OllamaResponse = response
            .json()
            .await
            .context("Failed to parse Ollama response")?;

        tracing::debug!("Ollama response: {}", ollama_response.response);

        Ok(ollama_response.response)
    }

    /// This is a real LLM backend, not simulated
    fn is_simulated(&self) -> bool {
        false
    }
}

/// Check if Ollama is available at the default endpoint
pub async fn check_ollama_availability() -> bool {
    let backend = OllamaBackend::new("gemma2:2b");
    backend.is_available().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_prompt_unknown() {
        let backend = OllamaBackend::new("test");
        let prompt = backend.build_system_prompt(&EpistemicStatus::Unknown);

        assert!(prompt.contains("do NOT know"));
        assert!(prompt.contains("MUST"));
        assert!(prompt.contains("DO NOT fabricate"));
    }

    #[test]
    fn test_system_prompt_uncertain() {
        let backend = OllamaBackend::new("test");
        let prompt = backend.build_system_prompt(&EpistemicStatus::Uncertain);

        assert!(prompt.contains("UNCERTAIN"));
        assert!(prompt.contains("uncertainty"));
    }

    #[test]
    fn test_system_prompt_known() {
        let backend = OllamaBackend::new("test");
        let prompt = backend.build_system_prompt(&EpistemicStatus::Known);

        assert!(prompt.contains("helpful"));
        assert!(prompt.contains("accurate"));
    }

    #[tokio::test]
    #[ignore] // Requires running Ollama
    async fn test_ollama_availability() {
        let available = check_ollama_availability().await;
        println!("Ollama available: {}", available);
    }

    #[tokio::test]
    #[ignore] // Requires running Ollama
    async fn test_ollama_unknown_refuses() {
        let backend = OllamaBackend::new("gemma2:2b");

        if !backend.is_available().await {
            println!("Skipping: Ollama not available");
            return;
        }

        let response = backend
            .generate("What is the GDP of Atlantis?", &EpistemicStatus::Unknown)
            .await
            .unwrap();

        let text = response.to_lowercase();

        // The response MUST contain hedging language
        let hedges = [
            "don't know", "do not know", "cannot", "no information",
            "unable to", "not available", "unknown",
        ];

        let contains_hedge = hedges.iter().any(|h| text.contains(h));
        assert!(
            contains_hedge,
            "LLM should refuse to answer but said: {}",
            response
        );
    }
}
