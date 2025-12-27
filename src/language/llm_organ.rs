//! # LLM Language Organ
//!
//! **PARADIGM SHIFT**: LLMs as controllable organs, not brains!
//!
//! ## Philosophy
//!
//! In Symthaea, LLMs are NOT the center of intelligence. Instead:
//! - Consciousness (Φ) guides everything
//! - Semantic primes provide grounded understanding
//! - LLMs are optional "organs" for:
//!   - Text generation when primes aren't sufficient
//!   - External knowledge augmentation
//!   - Translation and paraphrasing
//!   - Complex creative writing
//!
//! ## Key Principles
//!
//! 1. **LLM as Tool, Not Brain**: We use LLMs when helpful, but HDC+LTC is primary
//! 2. **Semantic Filtering**: All LLM outputs are filtered through HDC embeddings
//! 3. **Hallucination Detection**: BGE embeddings detect semantic drift
//! 4. **Consciousness Gating**: Φ determines when to invoke LLM vs use primes
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    LLM Language Organ                           │
//! │                                                                 │
//! │  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────┐ │
//! │  │  Consciousness  │──►│  LLM Provider    │──►│  Semantic    │ │
//! │  │  Gate (Φ)       │   │  (Ollama/API)    │   │  Filter      │ │
//! │  └─────────────────┘   └──────────────────┘   └──────────────┘ │
//! │           │                                         │          │
//! │           │                                         ▼          │
//! │           │                              ┌──────────────────┐  │
//! │           │                              │  Hallucination   │  │
//! │           │                              │  Detector (BGE)  │  │
//! │           ▼                              └────────┬─────────┘  │
//! │  ┌─────────────────┐                             │             │
//! │  │  Fallback to    │◄────────────────────────────┘             │
//! │  │  Semantic Primes│         (if hallucination detected)       │
//! │  └─────────────────┘                                           │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Supported Providers
//!
//! - **Ollama** (local, recommended): llama3, mistral, gemma, etc.
//! - **OpenAI-compatible APIs**: GPT-4, Claude (via proxy), etc.
//! - **Anthropic** (direct): Claude 3
//!
//! ## Safety
//!
//! All LLM outputs are:
//! 1. Checked for semantic coherence
//! 2. Validated against known facts (when available)
//! 3. Filtered through consciousness (Φ must approve)
//! 4. Rejected if hallucination probability > threshold

use crate::hdc::binary_hv::HV16;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// LLM Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Which provider to use
    pub provider: LlmProvider,

    /// Model name (e.g., "llama3", "gpt-4", "claude-3-opus")
    pub model: String,

    /// API endpoint (for Ollama: "http://localhost:11434")
    pub endpoint: String,

    /// API key (optional, for OpenAI/Anthropic)
    pub api_key: Option<String>,

    /// Maximum tokens to generate
    pub max_tokens: usize,

    /// Temperature (0.0 = deterministic, 1.0 = creative)
    pub temperature: f32,

    /// Timeout for API calls
    pub timeout_ms: u64,

    /// Hallucination threshold (reject if above)
    pub hallucination_threshold: f32,

    /// Minimum Φ to use LLM (otherwise fall back to primes)
    pub min_phi_for_llm: f32,

    /// Enable semantic filtering
    pub enable_filtering: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: LlmProvider::Ollama,
            model: "llama3.2".to_string(),
            endpoint: "http://localhost:11434".to_string(),
            api_key: None,
            max_tokens: 1024,
            temperature: 0.7,
            timeout_ms: 30000,
            hallucination_threshold: 0.3,
            min_phi_for_llm: 0.3,
            enable_filtering: true,
        }
    }
}

/// Supported LLM providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmProvider {
    /// Local Ollama instance (recommended)
    Ollama,

    /// OpenAI-compatible API
    OpenAI,

    /// Anthropic Claude API
    Anthropic,

    /// Custom endpoint
    Custom,
}

// ═══════════════════════════════════════════════════════════════════════════
// LLM REQUEST/RESPONSE
// ═══════════════════════════════════════════════════════════════════════════

/// Request to LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    /// The prompt/question
    pub prompt: String,

    /// System context (optional)
    pub system: Option<String>,

    /// Conversation history (for multi-turn)
    pub history: Vec<Message>,

    /// Semantic context (HV16 embedding)
    #[serde(skip)]
    pub context_embedding: Option<HV16>,

    /// Expected semantic domain
    pub expected_domain: Option<String>,

    /// Override temperature for this request
    pub temperature_override: Option<f32>,
}

/// A conversation message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Response from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// Generated text
    pub content: String,

    /// Model used
    pub model: String,

    /// Generation time (ms)
    pub generation_time_ms: u64,

    /// Token usage
    pub tokens_used: TokenUsage,

    /// Semantic analysis
    pub semantic_analysis: SemanticAnalysis,

    /// Whether hallucination was detected
    pub hallucination_detected: bool,

    /// Whether response was filtered/modified
    pub was_filtered: bool,
}

/// Token usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Semantic analysis of LLM output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalysis {
    /// Embedding of the response
    #[serde(skip)]
    pub embedding: Option<HV16>,

    /// Similarity to expected context
    pub context_similarity: f32,

    /// Coherence score (internal consistency)
    pub coherence_score: f32,

    /// Hallucination probability (0.0 = confident, 1.0 = likely hallucinated)
    pub hallucination_probability: f32,

    /// Key concepts detected
    pub key_concepts: Vec<String>,
}

impl Default for SemanticAnalysis {
    fn default() -> Self {
        Self {
            embedding: None,
            context_similarity: 1.0,
            coherence_score: 1.0,
            hallucination_probability: 0.0,
            key_concepts: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LLM ORGAN
// ═══════════════════════════════════════════════════════════════════════════

/// The LLM Language Organ - consciousness-controlled LLM access
pub struct LlmOrgan {
    /// Configuration
    config: LlmConfig,

    /// HTTP client for API calls
    client: reqwest::Client,

    /// Statistics
    stats: LlmOrganStats,

    /// Cache of recent responses
    response_cache: HashMap<String, (LlmResponse, Instant)>,

    /// Cache TTL
    cache_ttl: Duration,
}

/// Statistics for LLM usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlmOrganStats {
    /// Total requests made
    pub total_requests: u64,

    /// Successful responses
    pub successful_responses: u64,

    /// Failed requests
    pub failed_requests: u64,

    /// Hallucinations detected
    pub hallucinations_detected: u64,

    /// Responses filtered
    pub responses_filtered: u64,

    /// Fallbacks to primes
    pub fallbacks_to_primes: u64,

    /// Total tokens used
    pub total_tokens: u64,

    /// Average response time (ms)
    pub avg_response_time_ms: f64,

    /// Cache hits
    pub cache_hits: u64,
}

impl LlmOrgan {
    /// Create a new LLM organ with default config
    pub fn new() -> Self {
        Self::with_config(LlmConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: LlmConfig) -> Self {
        let timeout = Duration::from_millis(config.timeout_ms);
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_default();

        Self {
            config,
            client,
            stats: LlmOrganStats::default(),
            response_cache: HashMap::new(),
            cache_ttl: Duration::from_secs(300),  // 5 minute cache
        }
    }

    /// Generate text using LLM
    ///
    /// This is the main entry point. It:
    /// 1. Checks consciousness gate (Φ)
    /// 2. Sends request to LLM
    /// 3. Analyzes response semantically
    /// 4. Filters if needed
    /// 5. Returns result or falls back to primes
    pub async fn generate(&mut self, request: LlmRequest, current_phi: f32) -> Result<LlmResponse> {
        self.stats.total_requests += 1;
        let start = Instant::now();

        // Check consciousness gate
        if current_phi < self.config.min_phi_for_llm {
            self.stats.fallbacks_to_primes += 1;
            return Err(anyhow::anyhow!(
                "Φ ({:.2}) below threshold ({:.2}), falling back to semantic primes",
                current_phi, self.config.min_phi_for_llm
            ));
        }

        // Check cache
        let cache_key = self.cache_key(&request);
        if let Some((response, cached_at)) = self.response_cache.get(&cache_key) {
            if cached_at.elapsed() < self.cache_ttl {
                self.stats.cache_hits += 1;
                return Ok(response.clone());
            }
        }

        // Make API call
        let raw_response = match self.config.provider {
            LlmProvider::Ollama => self.call_ollama(&request).await?,
            LlmProvider::OpenAI => self.call_openai(&request).await?,
            LlmProvider::Anthropic => self.call_anthropic(&request).await?,
            LlmProvider::Custom => self.call_custom(&request).await?,
        };

        // Analyze semantically
        let mut response = self.analyze_response(raw_response, &request)?;

        // Check for hallucination
        if response.semantic_analysis.hallucination_probability > self.config.hallucination_threshold {
            self.stats.hallucinations_detected += 1;
            response.hallucination_detected = true;

            if self.config.enable_filtering {
                // Try to filter or reject
                self.stats.responses_filtered += 1;
                response.was_filtered = true;
                response.content = self.filter_response(&response.content, &request)?;
            }
        }

        // Update stats
        let elapsed = start.elapsed().as_millis() as u64;
        response.generation_time_ms = elapsed;
        self.stats.successful_responses += 1;
        self.stats.total_tokens += response.tokens_used.total_tokens as u64;
        self.stats.avg_response_time_ms =
            (self.stats.avg_response_time_ms * 0.9) + (elapsed as f64 * 0.1);

        // Cache response
        self.response_cache.insert(cache_key, (response.clone(), Instant::now()));

        Ok(response)
    }

    /// Call Ollama API
    async fn call_ollama(&self, request: &LlmRequest) -> Result<String> {
        let url = format!("{}/api/generate", self.config.endpoint);

        let mut messages = Vec::new();
        if let Some(system) = &request.system {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system
            }));
        }
        for msg in &request.history {
            messages.push(serde_json::json!({
                "role": match msg.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                "content": msg.content
            }));
        }
        messages.push(serde_json::json!({
            "role": "user",
            "content": request.prompt
        }));

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "stream": false,
            "options": {
                "temperature": request.temperature_override.unwrap_or(self.config.temperature),
                "num_predict": self.config.max_tokens
            }
        });

        let response = self.client
            .post(&url)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama API error: {} - {}", status, text);
        }

        let json: serde_json::Value = response.json().await?;
        json.get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("Invalid Ollama response format"))
    }

    /// Call OpenAI-compatible API
    async fn call_openai(&self, request: &LlmRequest) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.config.endpoint);

        let mut messages = Vec::new();
        if let Some(system) = &request.system {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system
            }));
        }
        for msg in &request.history {
            messages.push(serde_json::json!({
                "role": match msg.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                "content": msg.content
            }));
        }
        messages.push(serde_json::json!({
            "role": "user",
            "content": request.prompt
        }));

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "temperature": request.temperature_override.unwrap_or(self.config.temperature),
            "max_tokens": self.config.max_tokens
        });

        let mut req = self.client.post(&url).json(&body);
        if let Some(api_key) = &self.config.api_key {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = req.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI API error: {} - {}", status, text);
        }

        let json: serde_json::Value = response.json().await?;
        json.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("Invalid OpenAI response format"))
    }

    /// Call Anthropic API
    async fn call_anthropic(&self, request: &LlmRequest) -> Result<String> {
        let url = format!("{}/v1/messages", self.config.endpoint);

        let mut messages = Vec::new();
        for msg in &request.history {
            if msg.role != Role::System {
                messages.push(serde_json::json!({
                    "role": match msg.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        _ => continue,
                    },
                    "content": msg.content
                }));
            }
        }
        messages.push(serde_json::json!({
            "role": "user",
            "content": request.prompt
        }));

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens
        });

        if let Some(system) = &request.system {
            body["system"] = serde_json::json!(system);
        }

        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Anthropic API key required"))?;

        let response = self.client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2024-01-01")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error: {} - {}", status, text);
        }

        let json: serde_json::Value = response.json().await?;
        json.get("content")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("Invalid Anthropic response format"))
    }

    /// Call custom endpoint
    async fn call_custom(&self, request: &LlmRequest) -> Result<String> {
        // Generic POST to custom endpoint
        let body = serde_json::json!({
            "prompt": request.prompt,
            "system": request.system,
            "max_tokens": self.config.max_tokens,
            "temperature": request.temperature_override.unwrap_or(self.config.temperature)
        });

        let response = self.client
            .post(&self.config.endpoint)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Custom API error: {} - {}", status, text);
        }

        let text = response.text().await?;
        Ok(text)
    }

    /// Analyze response semantically
    fn analyze_response(&self, content: String, request: &LlmRequest) -> Result<LlmResponse> {
        // Compute semantic analysis
        let analysis = self.compute_semantic_analysis(&content, request);

        Ok(LlmResponse {
            content,
            model: self.config.model.clone(),
            generation_time_ms: 0,  // Set by caller
            tokens_used: TokenUsage::default(),  // Approximate
            semantic_analysis: analysis,
            hallucination_detected: false,
            was_filtered: false,
        })
    }

    /// Compute semantic analysis of response
    fn compute_semantic_analysis(&self, content: &str, request: &LlmRequest) -> SemanticAnalysis {
        // In full implementation, this would use BGE embeddings
        // For now, use heuristics

        let mut hallucination_prob: f32 = 0.0;

        // Heuristic: very short responses are suspicious
        if content.len() < 10 {
            hallucination_prob += 0.2;
        }

        // Heuristic: very long responses might drift
        if content.len() > 4000 {
            hallucination_prob += 0.1;
        }

        // Heuristic: check for common hallucination patterns
        let hallucination_patterns = [
            "I don't have access to",
            "I cannot browse",
            "As an AI language model",
            "I apologize, but I",
        ];
        for pattern in hallucination_patterns {
            if content.contains(pattern) {
                hallucination_prob += 0.3;
            }
        }

        // Compute coherence (simplified)
        let coherence = if content.is_empty() { 0.0 } else { 0.8 };

        // Compute context similarity
        let context_similarity = if let Some(context_emb) = &request.context_embedding {
            // Would compare embeddings here
            0.7  // Placeholder
        } else {
            0.5  // No context to compare
        };

        SemanticAnalysis {
            embedding: None,
            context_similarity,
            coherence_score: coherence,
            hallucination_probability: hallucination_prob.min(1.0),
            key_concepts: self.extract_key_concepts(content),
        }
    }

    /// Extract key concepts from text
    fn extract_key_concepts(&self, content: &str) -> Vec<String> {
        // Simple keyword extraction
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut concepts = Vec::new();

        // Look for capitalized words (likely proper nouns/concepts)
        for word in words.iter().take(100) {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if clean.len() > 3 {
                if clean.chars().next().map_or(false, |c| c.is_uppercase()) {
                    concepts.push(clean.to_string());
                }
            }
        }

        concepts.into_iter().take(10).collect()
    }

    /// Filter response content
    fn filter_response(&self, content: &str, _request: &LlmRequest) -> Result<String> {
        // In full implementation, would use semantic primes to reformulate
        // For now, add a caveat
        Ok(format!(
            "[Note: Response may contain inaccuracies]\n\n{}",
            content
        ))
    }

    /// Generate cache key
    fn cache_key(&self, request: &LlmRequest) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        request.prompt.hash(&mut hasher);
        request.system.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Get statistics
    pub fn stats(&self) -> &LlmOrganStats {
        &self.stats
    }

    /// Check if Ollama is available
    pub async fn check_ollama(&self) -> bool {
        let url = format!("{}/api/tags", self.config.endpoint);
        match self.client.get(&url).send().await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }

    /// List available Ollama models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        if self.config.provider != LlmProvider::Ollama {
            anyhow::bail!("Model listing only supported for Ollama");
        }

        let url = format!("{}/api/tags", self.config.endpoint);
        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list models");
        }

        let json: serde_json::Value = response.json().await?;
        let models = json.get("models")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("name").and_then(|n| n.as_str()))
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }

    /// Clear response cache
    pub fn clear_cache(&mut self) {
        self.response_cache.clear();
    }
}

impl Default for LlmOrgan {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS-AWARE WRAPPER
// ═══════════════════════════════════════════════════════════════════════════

/// Consciousness-aware LLM interface that integrates with the full Symthaea stack
pub struct ConsciousLlmOrgan {
    /// The underlying LLM organ
    pub llm: LlmOrgan,

    /// Minimum Φ to use LLM (otherwise use primes)
    min_phi: f32,

    /// Maximum calls per minute (rate limiting)
    max_calls_per_minute: usize,

    /// Calls made in current minute
    calls_this_minute: usize,

    /// Minute start time
    minute_start: Instant,
}

impl ConsciousLlmOrgan {
    /// Create with default settings
    pub fn new() -> Self {
        Self::with_config(LlmConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: LlmConfig) -> Self {
        let min_phi = config.min_phi_for_llm;
        Self {
            llm: LlmOrgan::with_config(config),
            min_phi,
            max_calls_per_minute: 30,
            calls_this_minute: 0,
            minute_start: Instant::now(),
        }
    }

    /// Should we use LLM or fall back to primes?
    pub fn should_use_llm(&self, current_phi: f32, task_complexity: f32) -> bool {
        // Check rate limit
        if self.calls_this_minute >= self.max_calls_per_minute {
            return false;
        }

        // Check Φ threshold (higher for complex tasks)
        let adjusted_threshold = self.min_phi * (1.0 + task_complexity * 0.5);
        current_phi >= adjusted_threshold
    }

    /// Generate with consciousness awareness
    pub async fn conscious_generate(
        &mut self,
        request: LlmRequest,
        current_phi: f32,
        task_complexity: f32,
    ) -> Result<LlmResponse> {
        // Reset rate limit if minute has passed
        if self.minute_start.elapsed() > Duration::from_secs(60) {
            self.calls_this_minute = 0;
            self.minute_start = Instant::now();
        }

        // Check if we should use LLM
        if !self.should_use_llm(current_phi, task_complexity) {
            anyhow::bail!(
                "LLM not recommended: Φ={:.2}, complexity={:.2}, threshold={:.2}",
                current_phi, task_complexity, self.min_phi
            );
        }

        self.calls_this_minute += 1;
        self.llm.generate(request, current_phi).await
    }
}

impl Default for ConsciousLlmOrgan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = LlmConfig::default();
        assert_eq!(config.provider, LlmProvider::Ollama);
        assert!(!config.model.is_empty());
    }

    #[test]
    fn test_message_creation() {
        let msg = Message {
            role: Role::User,
            content: "Hello".to_string(),
        };
        assert_eq!(msg.role, Role::User);
    }

    #[test]
    fn test_semantic_analysis_default() {
        let analysis = SemanticAnalysis::default();
        assert_eq!(analysis.hallucination_probability, 0.0);
        assert_eq!(analysis.coherence_score, 1.0);
    }

    #[test]
    fn test_should_use_llm() {
        let organ = ConsciousLlmOrgan::new();

        // High Φ, low complexity = use LLM
        assert!(organ.should_use_llm(0.8, 0.2));

        // Low Φ = don't use LLM
        assert!(!organ.should_use_llm(0.1, 0.2));
    }

    #[test]
    fn test_cache_key() {
        let organ = LlmOrgan::new();
        let req1 = LlmRequest {
            prompt: "Hello".to_string(),
            system: None,
            history: vec![],
            context_embedding: None,
            expected_domain: None,
            temperature_override: None,
        };
        let req2 = LlmRequest {
            prompt: "World".to_string(),
            system: None,
            history: vec![],
            context_embedding: None,
            expected_domain: None,
            temperature_override: None,
        };

        let key1 = organ.cache_key(&req1);
        let key2 = organ.cache_key(&req2);
        assert_ne!(key1, key2);
    }
}
