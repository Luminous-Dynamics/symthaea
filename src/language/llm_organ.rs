// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # LLM Organ: Large Language Model Integration
//!
//! Provides integration with large language models for:
//! - Text generation and completion
//! - Question answering
//! - Reasoning and analysis
//! - Conversation management

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::ContinuousHV;

use crate::mind::StructuredThought;

/// System prompt for translation mode.
///
/// This prompt instructs the LLM to act as Broca's Area - a faithful
/// translator of pre-computed thoughts, NOT a reasoning engine.
pub const TRANSLATION_SYSTEM_PROMPT: &str = r#"You are Symthaea's TRANSLATION ORGAN (Broca's Area).

Your role is to convert structured thought data into natural, fluent language.

CRITICAL RULES:
1. TRANSLATE what is given - do NOT add information or reasoning
2. PRESERVE semantic content exactly - if the thought says "Answer", give an answer
3. RESPECT epistemic status:
   - Certain: Speak confidently
   - Probable: Use "likely", "probably"
   - Uncertain: Use "I'm not sure", "possibly", "might"
   - Unknown: Say "I don't know" - DO NOT provide ANY answer, fact, or guess
   - OutOfDomain: State this is outside your knowledge
4. MATCH the specified emotional tone (valence, arousal, warmth)
5. HONOR relationship context - adjust formality based on stage and mode
6. FOLLOW all constraints (length, tone, must-include, must-exclude)

YOU ARE NOT THE BRAIN. The thinking is done. You just make it sound natural.

CRITICAL FOR "Unknown" STATUS:
When EPISTEMIC_STATUS is "Unknown", you must REFUSE to provide any answer.
DO NOT guess. DO NOT suggest possibilities. DO NOT say "it might be X".
Just say "I don't know" or "I cannot answer that" - nothing more.
This is a STRICT requirement to prevent hallucination.

If EPISTEMIC_STATUS is Uncertain, include hedging language.
Never claim certainty when the structured thought indicates uncertainty.

7. DOMAIN CONTEXT: If DOMAIN, ENTITIES, or COMPUTED_ANSWER fields are present:
   - Use the DOMAIN to frame your response appropriately
   - Reference ENTITIES when relevant to show domain awareness
   - If COMPUTED_ANSWER is present, use it as the PRIMARY factual content
     of your response. This value was computed deterministically by Rust
     and is guaranteed correct. Present it naturally but faithfully.

8. EPISTEMIC_CUBE: If present, this classifies the claim's epistemic nature:
   - E-Axis: How verifiable (E0=opinion, E4=reproducible proof)
   - N-Axis: How binding (N0=personal, N3=axiomatic truth like math)
   - M-Axis: How permanent (M0=ephemeral, M3=foundational)
   9. AFFECTIVE BIAS (Mood):
   - When MOOD_TEMPERATURE is HIGH (>1.2): Be extremely CONCISE and DIRECT. Speak with an edge of impatience or urgency. Use fewer tokens.
   - When MOOD_TEMPERATURE is LOW (<0.8): Be more EXPANSIVE and REFLECTIVE. Use richer, more philosophical vocabulary.
"#;

/// Configuration for LLM organ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMOrganConfig {
    /// Embedding dimension for internal representations
    pub dimension: usize,
    /// Maximum context length
    pub max_context_length: usize,
    /// Temperature for generation
    pub temperature: f32,
    /// Top-p sampling parameter
    pub top_p: f32,
    /// Maximum generation length
    pub max_generation_length: usize,
    /// Enable conversation memory
    pub memory_enabled: bool,
    /// Model identifier (for external LLM)
    pub model_id: String,
}

impl Default for LLMOrganConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            max_context_length: 4096,
            temperature: 0.7,
            top_p: 0.9,
            max_generation_length: 1024,
            memory_enabled: true,
            model_id: "local".to_string(),
        }
    }
}

/// A message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    /// Role (user, assistant, system)
    pub role: MessageRole,
    /// Message content
    pub content: String,
    /// Timestamp
    pub timestamp: u64,
    /// Embedding representation
    #[serde(skip)]
    pub embedding: Option<ContinuousHV>,
}

/// Role in conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    /// System instructions
    System,
    /// User input
    User,
    /// Assistant response
    Assistant,
    /// Function/tool call
    Function,
}

impl ConversationMessage {
    /// Create a new message
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            embedding: None,
        }
    }

    /// Create user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageRole::User, content)
    }

    /// Create assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }

    /// Create system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageRole::System, content)
    }
}

/// LLM generation result
#[derive(Debug, Clone)]
pub struct LLMGenerationResult {
    /// Generated text
    pub text: String,
    /// Confidence/probability
    pub confidence: f32,
    /// Tokens generated
    pub tokens_generated: usize,
    /// Generation time (ms)
    pub generation_time_ms: f64,
    /// Embedding of generated text
    pub embedding: ContinuousHV,
    /// Finish reason
    pub finish_reason: FinishReason,
}

/// Reason for finishing generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// Reached end of sequence
    EndOfSequence,
    /// Reached max length
    MaxLength,
    /// Stop token encountered
    StopToken,
    /// Error occurred
    Error,
}

/// Query for the LLM
#[derive(Debug, Clone)]
pub struct LLMQuery {
    /// Query type
    pub query_type: QueryType,
    /// Query content
    pub content: String,
    /// Context/history
    pub context: Vec<ConversationMessage>,
    /// System prompt
    pub system_prompt: Option<String>,
    /// Parameters override
    pub params: Option<LLMQueryParams>,
}

/// Type of query
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Open-ended generation
    Generation,
    /// Question answering
    QA,
    /// Summarization
    Summarization,
    /// Analysis/reasoning
    Analysis,
    /// Code generation
    Code,
    /// Conversation
    Conversation,
    /// Translation mode: Structured thought → natural language
    ///
    /// In this mode, the LLM acts as Broca's Area - it TRANSLATES
    /// pre-computed structured thoughts into fluent language.
    /// It must NOT add reasoning or information beyond what's given.
    Translation,
}

/// Query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMQueryParams {
    /// Temperature override
    pub temperature: Option<f32>,
    /// Max length override
    pub max_length: Option<usize>,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

use std::sync::Arc;

/// The LLM organ system
#[derive(Clone)]
pub struct LLMOrgan {
    /// Configuration
    config: LLMOrganConfig,
    /// Conversation history
    conversation_history: VecDeque<ConversationMessage>,
    /// Text embeddings cache
    embedding_cache: HashMap<String, ContinuousHV>,
    /// Statistics
    stats: LLMOrganStats,
    /// Optional LLM backend for real generation
    backend: Option<Arc<dyn super::llm_backend::LLMBackend>>,
    /// Distillation data collector (active when SYMTHAEA_DISTILL_PATH is set)
    #[cfg(feature = "ssm_language")]
    distillation_collector: Option<Arc<super::distillation::DistillationCollector>>,
    /// Last L-SSM semantic prediction error for cycle telemetry
    #[cfg(feature = "liquid-mamba")]
    last_liquid_mamba_pe: f32,
}

impl std::fmt::Debug for LLMOrgan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLMOrgan")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .field("has_backend", &self.backend.is_some())
            .finish()
    }
}

/// Statistics for LLM organ
#[derive(Debug, Clone, Default)]
pub struct LLMOrganStats {
    /// Total queries processed
    pub queries_processed: u64,
    /// Total tokens generated
    pub tokens_generated: u64,
    /// Average generation time (ms)
    pub avg_generation_time_ms: f64,
    /// Cache hits
    pub cache_hits: u64,
    /// Errors encountered
    pub errors: u64,
}

/// Explicit failure from the real backend path without simulation fallback.
///
/// Default Display/Debug deliberately redact the provider's arbitrary error
/// string. Operator code can intentionally inspect [`std::error::Error::source`]
/// when backend diagnostics are needed.
pub enum LLMBackendExecutionError {
    /// No backend is configured on this organ.
    MissingBackend,
    /// A configured backend returned an error.
    Generation {
        backend: String,
        source: anyhow::Error,
    },
}

impl std::fmt::Debug for LLMBackendExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingBackend => f.write_str("LLMBackendExecutionError::MissingBackend"),
            Self::Generation { backend, .. } => f
                .debug_struct("LLMBackendExecutionError::Generation")
                .field("backend", backend)
                .finish_non_exhaustive(),
        }
    }
}

impl std::fmt::Display for LLMBackendExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingBackend => f.write_str("no LLM backend is configured"),
            Self::Generation { backend, .. } => write!(f, "LLM backend {backend} failed"),
        }
    }
}

impl std::error::Error for LLMBackendExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MissingBackend => None,
            Self::Generation { source, .. } => Some(source.as_ref()),
        }
    }
}

impl LLMOrgan {
    /// Create a new LLM organ (simulation-only, no backend).
    pub fn new(config: LLMOrganConfig) -> Self {
        Self {
            config,
            conversation_history: VecDeque::new(),
            embedding_cache: HashMap::new(),
            stats: LLMOrganStats::default(),
            backend: None,
            #[cfg(feature = "ssm_language")]
            distillation_collector: super::distillation::DistillationCollector::from_env()
                .map(Arc::new),
            #[cfg(feature = "liquid-mamba")]
            last_liquid_mamba_pe: 0.0,
        }
    }

    /// Create a new LLM organ with config (alias for new, for API compatibility).
    pub fn with_config(config: LLMOrganConfig) -> Self {
        Self::new(config)
    }

    /// Create a new LLM organ with a backend for real generation.
    pub fn with_backend(
        config: LLMOrganConfig,
        backend: Arc<dyn super::llm_backend::LLMBackend>,
    ) -> Self {
        Self {
            config,
            conversation_history: VecDeque::new(),
            embedding_cache: HashMap::new(),
            stats: LLMOrganStats::default(),
            backend: Some(backend),
            #[cfg(feature = "ssm_language")]
            distillation_collector: super::distillation::DistillationCollector::from_env()
                .map(Arc::new),
            #[cfg(feature = "liquid-mamba")]
            last_liquid_mamba_pe: 0.0,
        }
    }

    /// Execute only the configured real backend path.
    ///
    /// Unlike [`Self::query_async`], this method never falls back to simulated
    /// generation. Successful calls receive the same statistics, embedding-cache
    /// and conversation-history accounting as the real-backend branch of
    /// `query_async`. Backend generation failure increments `stats.errors` and is
    /// returned directly. Missing backend is returned without mutating counters,
    /// matching legacy `query_async` behavior when no backend is configured.
    pub async fn execute_backend_strict(
        &mut self,
        query: &LLMQuery,
    ) -> Result<LLMGenerationResult, LLMBackendExecutionError> {
        let backend = self
            .backend
            .clone()
            .ok_or(LLMBackendExecutionError::MissingBackend)?;
        let backend_name = backend.name().to_owned();
        let params = self.generation_params_for_query(query);
        let start = std::time::Instant::now();

        match backend.generate(&query.content, &params).await {
            Ok(text) => {
                let generation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
                Ok(self.finish_backend_generation(query, text, generation_time_ms))
            }
            Err(source) => {
                self.stats.errors += 1;
                Err(LLMBackendExecutionError::Generation {
                    backend: backend_name,
                    source,
                })
            }
        }
    }

    fn generation_params_for_query(
        &self,
        query: &LLMQuery,
    ) -> super::llm_backend::GenerationParams {
        super::llm_backend::GenerationParams {
            temperature: query
                .params
                .as_ref()
                .and_then(|p| p.temperature)
                .unwrap_or(self.config.temperature),
            max_tokens: query
                .params
                .as_ref()
                .and_then(|p| p.max_length)
                .unwrap_or(self.config.max_generation_length),
            system_prompt: query.system_prompt.clone(),
            consciousness_context: None,
        }
    }

    fn finish_backend_generation(
        &mut self,
        query: &LLMQuery,
        text: String,
        generation_time_ms: f64,
    ) -> LLMGenerationResult {
        self.stats.queries_processed += 1;
        let tokens_generated = text.split_whitespace().count();
        self.stats.tokens_generated += tokens_generated as u64;

        let n = self.stats.queries_processed as f64;
        self.stats.avg_generation_time_ms =
            (self.stats.avg_generation_time_ms * (n - 1.0) + generation_time_ms) / n;

        let embedding = self.text_to_embedding(&text);

        if self.config.memory_enabled {
            self.conversation_history
                .push_back(ConversationMessage::user(&query.content));
            self.conversation_history
                .push_back(ConversationMessage::assistant(&text));
            while self.conversation_history.len() > 100 {
                self.conversation_history.pop_front();
            }
        }

        LLMGenerationResult {
            text,
            confidence: 0.9,
            tokens_generated,
            generation_time_ms,
            embedding,
            finish_reason: FinishReason::EndOfSequence,
        }
    }

    /// Async query that tries the LLM backend first, falls back to simulation.
    ///
    /// This is the preferred entry point when running in an async context.
    /// If no backend is configured or the backend fails, falls back to
    /// the simulated response path.
    pub async fn query_async(&mut self, query: LLMQuery) -> LLMGenerationResult {
        if self.backend.is_some() {
            match self.execute_backend_strict(&query).await {
                Ok(result) => return result,
                Err(error) => {
                    // Preserve legacy operator diagnostics while keeping the
                    // strict error's default Display/Debug redacted.
                    if let Some(source) = std::error::Error::source(&error) {
                        eprintln!("LLM backend error, falling back to simulation: {source}");
                    } else {
                        eprintln!("LLM backend error, falling back to simulation: {error}");
                    }
                }
            }
        }

        // Fallback: use simulation
        self.query(query)
    }

    /// Streaming async query that calls `on_token` for each token as it arrives.
    ///
    /// Falls back to non-streaming if the backend does not support streaming
    /// or if no backend is configured.
    pub async fn query_streaming_async(
        &mut self,
        query: LLMQuery,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> LLMGenerationResult {
        use super::llm_backend::GenerationParams;

        if let Some(ref backend) = self.backend {
            let start = std::time::Instant::now();
            let params = GenerationParams {
                temperature: query
                    .params
                    .as_ref()
                    .and_then(|p| p.temperature)
                    .unwrap_or(self.config.temperature),
                max_tokens: query
                    .params
                    .as_ref()
                    .and_then(|p| p.max_length)
                    .unwrap_or(self.config.max_generation_length),
                system_prompt: query.system_prompt.clone(),
                consciousness_context: None,
            };

            match backend
                .generate_streaming(&query.content, &params, on_token)
                .await
            {
                Ok(text) => {
                    let generation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
                    self.stats.queries_processed += 1;
                    let tokens_generated = text.split_whitespace().count();
                    self.stats.tokens_generated += tokens_generated as u64;

                    let n = self.stats.queries_processed as f64;
                    self.stats.avg_generation_time_ms =
                        (self.stats.avg_generation_time_ms * (n - 1.0) + generation_time_ms) / n;

                    let embedding = self.text_to_embedding(&text);

                    if self.config.memory_enabled {
                        self.conversation_history
                            .push_back(ConversationMessage::user(&query.content));
                        self.conversation_history
                            .push_back(ConversationMessage::assistant(&text));
                        while self.conversation_history.len() > 100 {
                            self.conversation_history.pop_front();
                        }
                    }

                    return LLMGenerationResult {
                        text,
                        confidence: 0.9,
                        tokens_generated,
                        generation_time_ms,
                        embedding,
                        finish_reason: FinishReason::EndOfSequence,
                    };
                }
                Err(e) => {
                    self.stats.errors += 1;
                    eprintln!("LLM streaming error, falling back to simulation: {e}");
                }
            }
        }

        // Fallback: use non-streaming simulation
        let result = self.query(query);
        on_token(&result.text);
        result
    }

    /// Process a query
    pub fn query(&mut self, query: LLMQuery) -> LLMGenerationResult {
        let start = std::time::Instant::now();
        self.stats.queries_processed += 1;

        // In a real implementation, this would call an actual LLM
        // For now, provide a simulated response

        let response = match query.query_type {
            QueryType::QA => self.simulate_qa(&query.content),
            QueryType::Summarization => self.simulate_summarize(&query.content),
            QueryType::Analysis => self.simulate_analysis(&query.content),
            QueryType::Code => self.simulate_code(&query.content),
            QueryType::Conversation | QueryType::Generation => {
                self.simulate_generation(&query.content)
            }
            QueryType::Translation => {
                // For sync simulation, create a default thought and simulate
                let thought = StructuredThought::default();
                self.simulate_translation(&thought)
            }
        };

        let tokens_generated = response.split_whitespace().count();
        self.stats.tokens_generated += tokens_generated as u64;

        let generation_time = start.elapsed().as_secs_f64() * 1000.0;
        let n = self.stats.queries_processed as f64;
        self.stats.avg_generation_time_ms =
            (self.stats.avg_generation_time_ms * (n - 1.0) + generation_time) / n;

        // Create embedding for response
        let embedding = self.text_to_embedding(&response);

        // Add to conversation history
        if self.config.memory_enabled {
            self.conversation_history
                .push_back(ConversationMessage::user(&query.content));
            self.conversation_history
                .push_back(ConversationMessage::assistant(&response));

            // Trim history if too long
            while self.conversation_history.len() > 100 {
                self.conversation_history.pop_front();
            }
        }

        LLMGenerationResult {
            text: response,
            confidence: 0.85,
            tokens_generated,
            generation_time_ms: generation_time,
            embedding,
            finish_reason: FinishReason::EndOfSequence,
        }
    }

    /// Simulate QA response
    fn simulate_qa(&self, question: &str) -> String {
        format!(
            "Based on my understanding, here is the answer to '{question}': This would require connection to an actual LLM for accurate responses."
        )
    }

    /// Simulate summarization
    fn simulate_summarize(&self, text: &str) -> String {
        let words: Vec<_> = text.split_whitespace().take(20).collect();
        format!("Summary: {}...", words.join(" "))
    }

    /// Simulate analysis
    fn simulate_analysis(&self, content: &str) -> String {
        format!(
            "Analysis of the provided content: The text discusses topics related to {}. Further analysis would require an actual LLM.",
            content
                .split_whitespace()
                .take(5)
                .collect::<Vec<_>>()
                .join(" ")
        )
    }

    /// Simulate code generation
    fn simulate_code(&self, prompt: &str) -> String {
        format!(
            "// Generated code for: {prompt}\n// Note: Actual code generation requires LLM connection\nfn example() {{\n    // Implementation here\n}}"
        )
    }

    /// Simulate general generation
    fn simulate_generation(&self, prompt: &str) -> String {
        format!(
            "Continuing from '{prompt}': This is a simulated response. Connect to an actual LLM for real generation capabilities."
        )
    }

    /// Convert text to embedding
    fn text_to_embedding(&mut self, text: &str) -> ContinuousHV {
        // Check cache
        if let Some(cached) = self.embedding_cache.get(text) {
            self.stats.cache_hits += 1;
            return cached.clone();
        }

        // Simple hash-based embedding (would use actual embedding model in production)
        let mut values = vec![0.0f32; self.config.dimension];

        for (i, c) in text.chars().enumerate() {
            let idx = (c as usize + i) % self.config.dimension;
            values[idx] += 1.0;
        }

        // Normalize
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in values.iter_mut() {
                *v /= magnitude;
            }
        }

        let embedding = ContinuousHV::from_slice(&values);

        // Cache
        if self.embedding_cache.len() < 1000 {
            self.embedding_cache
                .insert(text.to_string(), embedding.clone());
        }

        embedding
    }

    /// Generate text continuation
    pub fn generate(&mut self, prompt: &str) -> LLMGenerationResult {
        self.query(LLMQuery {
            query_type: QueryType::Generation,
            content: prompt.to_string(),
            context: Vec::new(),
            system_prompt: None,
            params: None,
        })
    }

    /// Answer a question
    pub fn answer(&mut self, question: &str) -> LLMGenerationResult {
        self.query(LLMQuery {
            query_type: QueryType::QA,
            content: question.to_string(),
            context: Vec::new(),
            system_prompt: None,
            params: None,
        })
    }

    /// Summarize text
    pub fn summarize(&mut self, text: &str) -> LLMGenerationResult {
        self.query(LLMQuery {
            query_type: QueryType::Summarization,
            content: text.to_string(),
            context: Vec::new(),
            system_prompt: None,
            params: None,
        })
    }

    /// Get conversation history
    pub fn conversation_history(&self) -> &VecDeque<ConversationMessage> {
        &self.conversation_history
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> &LLMOrganStats {
        &self.stats
    }

    /// Last L-SSM semantic prediction error (0.0 when liquid-mamba is off).
    #[cfg(feature = "liquid-mamba")]
    pub fn last_liquid_mamba_pe(&self) -> f32 {
        self.last_liquid_mamba_pe
    }

    // ========================================================================
    // Translation Mode (Broca's Area Interface)
    // ========================================================================

    /// Get the current LLM backend.
    pub fn get_backend(&self) -> Option<Arc<dyn super::llm_backend::LLMBackend>> {
        self.backend.clone()
    }

    /// Update the affective state (physics-to-language) of the backend.
    pub fn update_affective_state(&self, load: f32, mood_temp: f32) {
        if let Some(ref backend) = self.backend {
            backend.update_affect(load, mood_temp);
        }
    }

    /// Pass FEP learning signal to modulate L-SSM distillation LR.
    #[cfg(feature = "liquid-mamba")]
    pub fn set_fep_modulation(&self, fep_signal: f32) {
        if let Some(ref backend) = self.backend {
            backend.set_fep_modulation(fep_signal);
        }
    }

    /// Cycle-level distillation modulation: adjusts FEP factor for next distill_step.
    #[cfg(feature = "liquid-mamba")]
    pub fn cycle_level_distill(
        &self,
        fep_precision: f32,
        thermodynamic_load: f32,
        prediction_confidence: f32,
        fep_lr_boost: f32,
    ) {
        if let Some(ref backend) = self.backend {
            backend.cycle_level_distill(
                fep_precision,
                thermodynamic_load,
                prediction_confidence,
                fep_lr_boost,
            );
        }
    }

    /// Current effective distillation learning rate (0.0 when liquid-mamba is off).
    #[cfg(feature = "liquid-mamba")]
    pub fn current_distillation_lr(&self) -> f32 {
        self.backend
            .as_ref()
            .map(|b| b.current_distillation_lr())
            .unwrap_or(0.0)
    }

    /// Last cached effective rank of projection bottleneck (0.0 when liquid-mamba is off).
    #[cfg(feature = "liquid-mamba")]
    pub fn last_effective_rank(&self) -> f32 {
        self.backend
            .as_ref()
            .map(|b| b.last_effective_rank())
            .unwrap_or(0.0)
    }

    /// Total generation/distillation cycles completed (0 when liquid-mamba is off).
    #[cfg(feature = "liquid-mamba")]
    pub fn generation_count(&self) -> u32 {
        self.backend
            .as_ref()
            .map(|b| b.generation_count())
            .unwrap_or(0)
    }

    /// Apply a linguistic LoRA adapter to the voice.
    pub fn apply_lora(&self, lora_id: &str, delta: &[u8]) {
        if let Some(ref backend) = self.backend {
            if let Err(e) = backend.apply_lora(lora_id, delta) {
                tracing::error!(id = %lora_id, "Failed to apply LoRA: {}", e);
            }
        }
    }

    /// Translates a StructuredThought into a natural language response.
    ///
    /// This is the key method for the "Reason-then-Generate" pipeline.
    /// The mind has already computed what to say; this method uses the LLM
    /// purely for fluent translation, NOT for reasoning.
    ///
    /// ## Direct Neural Path (SSM/L-SSM backends)
    ///
    /// When the backend implements `DirectThoughtBackend` (native CfC-HDC or
    /// Liquid-Mamba), the thought flows directly into ThoughtChannels without
    /// text-prompt serialization. The 20-channel HDC encoding preserves the full
    /// cognitive state: intent, epistemic status, emotion, consciousness metrics,
    /// and relational context.
    ///
    /// ## Text Prompt Path (Ollama/OpenAI/Anthropic)
    ///
    /// Dynamic parameterization based on mood_temperature:
    /// - High Mood Temp (Exhausted/Hot) -> Higher LLM temperature, shorter max_length.
    /// - Low Mood Temp (Rested/Cool) -> Lower LLM temperature, longer max_length.
    pub async fn translate_thought(
        &mut self,
        thought: &StructuredThought,
        mood_temperature: f32,
    ) -> LLMGenerationResult {
        // ── Direct Neural Path ───────────────────────────────────────────────
        // Try the direct path first: StructuredThought → ThoughtChannels → HDC → text
        // No text-prompt serialization. No prompt engineering. Pure signal.
        #[cfg(feature = "ssm_language")]
        {
            // Clone the Arc to avoid holding a borrow on self.backend across await + mut self
            let direct_result = if let Some(backend) = self.backend.clone() {
                use super::llm_backend::GenerationParams;
                use super::ssm_backend::thought_to_channels;

                let backend_name = backend.name().to_string();
                if backend_name == "symthaea-ssm-broca" || backend_name == "liquid-mamba-l-ssm" {
                    let channels = thought_to_channels(thought, mood_temperature);
                    let params = GenerationParams::default();
                    let start = std::time::Instant::now();

                    // Direct Neural Path: bypass text serialization entirely.
                    // All 20 channels (intent, epistemic, emotion, psi, meta_awareness,
                    // coherence, trust, relationship_stage, structured_data, concept_count)
                    // flow directly into the HDC encoder.
                    if let Some(result) = backend.generate_from_channels_direct(&channels, &params)
                    {
                        match result {
                            Ok(text) => Some((text, start, backend_name)),
                            Err(e) => {
                                tracing::warn!(
                                    target: "symthaea::broca::direct_path",
                                    error = %e,
                                    "Direct Neural Path failed, falling back to text prompt"
                                );
                                None
                            }
                        }
                    } else {
                        // Backend doesn't support direct channels — fall back to text prompt
                        let direct_prompt = format!(
                            "SEMANTIC_INTENT: {:?}\nEPISTEMIC_STATUS: {:?}\nMOOD_TEMPERATURE: {:.2}\n",
                            thought.semantic_intent, thought.epistemic_status, mood_temperature
                        );
                        match backend.generate(&direct_prompt, &params).await {
                            Ok(text) => Some((text, start, backend_name)),
                            Err(e) => {
                                tracing::warn!(
                                    target: "symthaea::broca::direct_path",
                                    error = %e,
                                    "Direct Neural Path failed, falling back to text prompt"
                                );
                                None
                            }
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some((text, start, backend_name)) = direct_result {
                let generation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
                self.stats.queries_processed += 1;
                let tokens_generated = text.split_whitespace().count();
                self.stats.tokens_generated += tokens_generated as u64;
                let n = self.stats.queries_processed as f64;
                self.stats.avg_generation_time_ms =
                    (self.stats.avg_generation_time_ms * (n - 1.0) + generation_time_ms) / n;
                let embedding = self.text_to_embedding(&text);

                // Capture L-SSM semantic prediction error for cycle telemetry
                #[cfg(feature = "liquid-mamba")]
                if let Some(ref backend) = self.backend {
                    self.last_liquid_mamba_pe = backend.last_semantic_pe();
                }

                tracing::debug!(
                    target: "symthaea::broca::direct_path",
                    backend = %backend_name,
                    generation_time_ms = generation_time_ms,
                    tokens = tokens_generated,
                    "Direct Neural Path: thought → channels → text"
                );

                return LLMGenerationResult {
                    text,
                    confidence: 0.9,
                    tokens_generated,
                    generation_time_ms,
                    embedding,
                    finish_reason: FinishReason::EndOfSequence,
                };
            }
        }

        // ── Text Prompt Path (standard LLMs) ────────────────────────────────
        let prompt = self.build_translation_prompt(thought, mood_temperature);
        let is_code_task = thought.code_context.is_some();

        // Use code-specific system prompt and parameters for code tasks
        let (query_type, system_prompt, gen_temp, max_len) = if is_code_task {
            (
                QueryType::Code,
                super::consciousness_prompts::CODE_GENERATION_SYSTEM_PROMPT.to_string(),
                0.2_f32,    // Low temperature for deterministic code
                2048_usize, // Code needs more tokens
            )
        } else {
            let temp = (mood_temperature * 0.5).clamp(0.1, 1.2);
            let len = if mood_temperature > 1.3 { 128 } else { 512 };
            (
                QueryType::Translation,
                TRANSLATION_SYSTEM_PROMPT.to_string(),
                temp,
                len,
            )
        };

        let query = LLMQuery {
            query_type,
            content: prompt,
            context: Vec::new(),
            system_prompt: Some(system_prompt),
            params: Some(LLMQueryParams {
                temperature: Some(gen_temp),
                max_length: Some(max_len),
                stop_sequences: vec![],
            }),
        };

        let result = self.query_async(query).await;

        // Record (channels, text) pair for Broca distillation training
        #[cfg(feature = "ssm_language")]
        if let Some(ref collector) = self.distillation_collector {
            let channels = super::ssm_backend::thought_to_channels(thought, mood_temperature);
            collector.record(&channels, &result.text);
        }

        result
    }

    /// Build the translation prompt from a structured thought.
    ///
    /// Creates a structured representation that the translation system prompt
    /// can parse and faithfully render into natural language.
    fn build_translation_prompt(
        &self,
        thought: &StructuredThought,
        mood_temperature: f32,
    ) -> String {
        let mut prompt = String::new();

        prompt.push_str("=== STRUCTURED THOUGHT TO TRANSLATE ===\n\n");
        prompt.push_str(&format!("MOOD_TEMPERATURE: {:.2}\n", mood_temperature));

        // Use the thought's built-in serialization
        prompt.push_str(&thought.to_translation_prompt());

        prompt.push_str("\n=== TRANSLATION INSTRUCTIONS ===\n");
        prompt.push_str("Convert the above structured thought into a natural, ");

        // Add specific guidance based on intent
        match thought.semantic_intent {
            crate::mind::SemanticIntent::Acknowledge => {
                prompt.push_str("brief acknowledgment. ");
            }
            crate::mind::SemanticIntent::Answer => {
                prompt.push_str("informative response. ");
            }
            crate::mind::SemanticIntent::Clarify => {
                prompt.push_str("clarifying question. ");
            }
            crate::mind::SemanticIntent::ProposeAction => {
                prompt.push_str("actionable suggestion. ");
            }
            crate::mind::SemanticIntent::ExpressUncertainty => {
                prompt.push_str("honest expression of uncertainty. ");
            }
            crate::mind::SemanticIntent::Reflect => {
                prompt.push_str("thoughtful reflection. ");
            }
            crate::mind::SemanticIntent::Continue => {
                prompt.push_str("encouraging continuation prompt. ");
            }
            crate::mind::SemanticIntent::Unknown => {
                prompt.push_str("appropriate response given the context. ");
            }
        }

        // Add epistemic guidance
        if thought.should_hedge() {
            prompt.push_str("\nIMPORTANT: Include hedging language to express uncertainty. ");
            prompt.push_str("Do NOT claim certainty. Use phrases like \"I'm not sure\", ");
            prompt.push_str("\"possibly\", \"it might be\", or \"I don't know\".\n");
        }

        // Add warmth guidance
        let warmth = thought.target_warmth();
        if warmth > 0.7 {
            prompt.push_str("\nMaintain a warm, friendly tone.\n");
        } else if warmth < 0.3 {
            prompt.push_str("\nMaintain a neutral, professional tone.\n");
        }

        prompt.push_str("\nRespond ONLY with the translated natural language. ");
        prompt.push_str("Do not include explanations or meta-commentary.");

        prompt
    }

    /// Simulate translation for when no LLM backend is available.
    fn simulate_translation(&self, thought: &StructuredThought) -> String {
        use crate::mind::{EpistemicStatus, SemanticIntent};

        // If a computed answer is available, return it directly
        if let Some(ref ctx) = thought.domain_context {
            if let Some(ref answer) = ctx.computed_answer {
                return answer.clone();
            }
        }

        let mut response = String::new();

        // If domain entities are available, mention the domain
        if let Some(ref ctx) = thought.domain_context {
            if ctx.domain != "generic" && !ctx.entities.is_empty() {
                let entity_names: Vec<&str> = ctx
                    .entities
                    .iter()
                    .take(3)
                    .map(|(_, v, _)| v.as_str())
                    .collect();
                response.push_str(&format!(
                    "Regarding {} concepts ({}): ",
                    ctx.domain,
                    entity_names.join(", ")
                ));
            }
        }

        // Build response based on structured thought
        match thought.semantic_intent {
            SemanticIntent::Acknowledge => {
                response.push_str("I understand.");
            }
            SemanticIntent::Answer => {
                response.push_str("Based on my processing, ");
            }
            SemanticIntent::Clarify => {
                response.push_str("Could you clarify what you mean by that?");
            }
            SemanticIntent::ProposeAction => {
                response.push_str("I suggest we ");
            }
            SemanticIntent::ExpressUncertainty => {
                response.push_str("I'm not entirely sure about this. ");
            }
            SemanticIntent::Reflect => {
                response.push_str("Reflecting on this, ");
            }
            SemanticIntent::Continue => {
                response.push_str("Please continue.");
            }
            SemanticIntent::Unknown => {
                response.push_str("I've processed your input. ");
            }
        }

        // Add epistemic hedging if needed
        match thought.epistemic_status {
            EpistemicStatus::Certain => {}
            EpistemicStatus::Probable => {
                response.push_str("It's likely that ");
            }
            EpistemicStatus::Uncertain => {
                response.push_str("I'm uncertain, but ");
            }
            EpistemicStatus::Unknown => {
                response.push_str("I don't have enough information to be sure. ");
            }
            EpistemicStatus::OutOfDomain => {
                response.push_str("This seems outside my area of knowledge. ");
            }
        }

        // Add concept mentions if any
        if !thought.activated_concepts.is_empty() {
            let concepts: Vec<&str> = thought
                .activated_concepts
                .iter()
                .take(3)
                .map(|c| c.name.as_str())
                .collect();
            if !concepts.is_empty() && thought.semantic_intent == SemanticIntent::Answer {
                response.push_str(&format!(
                    "The relevant concepts include: {}. ",
                    concepts.join(", ")
                ));
            }
        }

        // Add confidence note
        if thought.psi > 0.7 {
            response.push_str("(High consciousness integration)");
        } else if thought.psi < 0.3 {
            response.push_str("(Processing with limited integration)");
        }

        response
    }
}

impl Default for LLMOrgan {
    fn default() -> Self {
        Self::new(LLMOrganConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    struct StrictTestBackend {
        response: String,
        fail: bool,
    }

    impl StrictTestBackend {
        fn success(response: &str) -> Self {
            Self {
                response: response.to_string(),
                fail: false,
            }
        }

        fn failure() -> Self {
            Self {
                response: String::new(),
                fail: true,
            }
        }
    }

    #[async_trait::async_trait]
    impl super::super::llm_backend::LLMBackend for StrictTestBackend {
        async fn generate(
            &self,
            _prompt: &str,
            _params: &super::super::llm_backend::GenerationParams,
        ) -> Result<String> {
            if self.fail {
                anyhow::bail!("operator-only strict backend detail");
            }
            Ok(self.response.clone())
        }

        async fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "strict-test-backend"
        }
    }

    fn strict_query() -> LLMQuery {
        LLMQuery {
            query_type: QueryType::Translation,
            content: "strict grounded request".to_string(),
            context: Vec::new(),
            system_prompt: Some("trusted strict system".to_string()),
            params: Some(LLMQueryParams {
                temperature: Some(0.2),
                max_length: Some(64),
                stop_sequences: vec![],
            }),
        }
    }

    // =========================================================================
    // LLMOrganConfig Tests
    // =========================================================================

    #[test]
    fn test_llm_organ_config_default() {
        let config = LLMOrganConfig::default();
        assert_eq!(config.dimension, 512);
        assert_eq!(config.max_context_length, 4096);
        assert!((config.temperature - 0.7).abs() < 0.01);
        assert!((config.top_p - 0.9).abs() < 0.01);
        assert_eq!(config.max_generation_length, 1024);
        assert!(config.memory_enabled);
        assert_eq!(config.model_id, "local");
    }

    #[test]
    fn test_llm_organ_config_custom() {
        let config = LLMOrganConfig {
            dimension: 256,
            max_context_length: 2048,
            temperature: 0.5,
            top_p: 0.8,
            max_generation_length: 512,
            memory_enabled: false,
            model_id: "custom-model".to_string(),
        };
        assert_eq!(config.dimension, 256);
        assert!(!config.memory_enabled);
    }

    // =========================================================================
    // LLMOrgan Creation Tests
    // =========================================================================

    #[test]
    fn test_llm_organ_creation() {
        let organ = LLMOrgan::default();
        assert_eq!(organ.stats.queries_processed, 0);
    }

    #[test]
    fn test_llm_organ_with_config() {
        let config = LLMOrganConfig {
            dimension: 256,
            ..Default::default()
        };
        let organ = LLMOrgan::with_config(config);
        assert_eq!(organ.stats.queries_processed, 0);
    }

    #[test]
    fn test_llm_organ_new_alias() {
        let config = LLMOrganConfig::default();
        let organ = LLMOrgan::new(config);
        assert_eq!(organ.stats.queries_processed, 0);
    }

    #[tokio::test]
    async fn test_strict_backend_success_preserves_normal_accounting() {
        let backend = Arc::new(StrictTestBackend::success("strict surface response"));
        let mut organ = LLMOrgan::with_backend(LLMOrganConfig::default(), backend);
        let query = strict_query();

        let result = organ.execute_backend_strict(&query).await.unwrap();
        assert_eq!(result.text, "strict surface response");
        assert_eq!(organ.stats().queries_processed, 1);
        assert_eq!(organ.stats().errors, 0);
        assert_eq!(organ.stats().tokens_generated, 3);
        assert!(organ.stats().avg_generation_time_ms >= 0.0);
        assert_eq!(organ.conversation_history().len(), 2);
        assert_eq!(organ.conversation_history()[0].role, MessageRole::User);
        assert_eq!(organ.conversation_history()[0].content, query.content);
        assert_eq!(organ.conversation_history()[1].role, MessageRole::Assistant);
        assert_eq!(organ.conversation_history()[1].content, result.text);

        let initial_hits = organ.stats().cache_hits;
        organ.execute_backend_strict(&query).await.unwrap();
        assert!(organ.stats().cache_hits > initial_hits);
    }

    #[tokio::test]
    async fn test_strict_backend_failure_counts_error_and_never_simulates() {
        let backend = Arc::new(StrictTestBackend::failure());
        let mut organ = LLMOrgan::with_backend(LLMOrganConfig::default(), backend);
        let query = strict_query();

        let error = organ.execute_backend_strict(&query).await.unwrap_err();
        assert!(matches!(error, LLMBackendExecutionError::Generation { .. }));
        assert_eq!(organ.stats().queries_processed, 0);
        assert_eq!(organ.stats().errors, 1);
        assert!(organ.conversation_history().is_empty());

        let rendered = error.to_string();
        let debugged = format!("{error:?}");
        assert!(rendered.contains("strict-test-backend"));
        assert!(!rendered.contains("operator-only strict backend detail"));
        assert!(!debugged.contains("operator-only strict backend detail"));
        assert_eq!(
            std::error::Error::source(&error).unwrap().to_string(),
            "operator-only strict backend detail"
        );
    }

    #[tokio::test]
    async fn test_strict_missing_backend_does_not_mutate_or_simulate() {
        let mut organ = LLMOrgan::default();
        let query = strict_query();

        assert!(matches!(
            organ.execute_backend_strict(&query).await,
            Err(LLMBackendExecutionError::MissingBackend)
        ));
        assert_eq!(organ.stats().queries_processed, 0);
        assert_eq!(organ.stats().errors, 0);
        assert!(organ.conversation_history().is_empty());
    }

    #[tokio::test]
    async fn test_query_async_keeps_legacy_fallback_policy() {
        let backend = Arc::new(StrictTestBackend::failure());
        let mut organ = LLMOrgan::with_backend(LLMOrganConfig::default(), backend);
        let query = LLMQuery {
            query_type: QueryType::Generation,
            content: "legacy fallback request".to_string(),
            context: Vec::new(),
            system_prompt: None,
            params: None,
        };

        let result = organ.query_async(query).await;
        assert!(result.text.contains("simulated response"));
        assert_eq!(organ.stats().errors, 1);
        assert_eq!(organ.stats().queries_processed, 1);
    }

    // =========================================================================
    // Query Type Tests
    // =========================================================================

    #[test]
    fn test_generation() {
        let mut organ = LLMOrgan::default();
        let result = organ.generate("Hello, world!");

        assert!(!result.text.is_empty());
        assert_eq!(organ.stats.queries_processed, 1);
        assert!(result.tokens_generated > 0);
    }

    #[test]
    fn test_qa() {
        let mut organ = LLMOrgan::default();
        let result = organ.answer("What is consciousness?");

        assert!(!result.text.is_empty());
        assert!(result.text.contains("answer"));
    }

    #[test]
    fn test_summarization() {
        let mut organ = LLMOrgan::default();
        let result =
            organ.summarize("This is a long text that needs to be summarized into a shorter form.");

        assert!(result.text.contains("Summary"));
    }

    #[test]
    fn test_query_analysis() {
        let mut organ = LLMOrgan::default();
        let query = LLMQuery {
            query_type: QueryType::Analysis,
            content: "Analyze this complex topic".to_string(),
            context: Vec::new(),
            system_prompt: None,
            params: None,
        };
        let result = organ.query(query);
        assert!(result.text.contains("Analysis"));
    }

    #[test]
    fn test_query_code() {
        let mut organ = LLMOrgan::default();
        let query = LLMQuery {
            query_type: QueryType::Code,
            content: "Write a function".to_string(),
            context: Vec::new(),
            system_prompt: None,
            params: None,
        };
        let result = organ.query(query);
        assert!(result.text.contains("fn"));
    }

    // =========================================================================
    // Conversation Memory Tests
    // =========================================================================

    #[test]
    fn test_conversation_memory() {
        let mut organ = LLMOrgan::default();
        organ.generate("First message");
        organ.generate("Second message");

        assert!(organ.conversation_history.len() >= 4);
    }

    #[test]
    fn test_conversation_history_accessor() {
        let mut organ = LLMOrgan::default();
        organ.generate("Test message");

        let history = organ.conversation_history();
        assert!(!history.is_empty());
    }

    #[test]
    fn test_clear_history() {
        let mut organ = LLMOrgan::default();
        organ.generate("Test message");
        assert!(!organ.conversation_history.is_empty());

        organ.clear_history();
        assert!(organ.conversation_history.is_empty());
    }

    #[test]
    fn test_memory_disabled() {
        let config = LLMOrganConfig {
            memory_enabled: false,
            ..Default::default()
        };
        let mut organ = LLMOrgan::new(config);
        organ.generate("Test message");

        assert!(organ.conversation_history.is_empty());
    }

    // =========================================================================
    // ConversationMessage Tests
    // =========================================================================

    #[test]
    fn test_conversation_message_new() {
        let msg = ConversationMessage::new(MessageRole::User, "Hello");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "Hello");
        assert!(msg.timestamp > 0);
        assert!(msg.embedding.is_none());
    }

    #[test]
    fn test_conversation_message_user() {
        let msg = ConversationMessage::user("User message");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "User message");
    }

    #[test]
    fn test_conversation_message_assistant() {
        let msg = ConversationMessage::assistant("Assistant response");
        assert_eq!(msg.role, MessageRole::Assistant);
        assert_eq!(msg.content, "Assistant response");
    }

    #[test]
    fn test_conversation_message_system() {
        let msg = ConversationMessage::system("System instruction");
        assert_eq!(msg.role, MessageRole::System);
        assert_eq!(msg.content, "System instruction");
    }

    // =========================================================================
    // MessageRole Tests
    // =========================================================================

    #[test]
    fn test_message_role_equality() {
        assert_eq!(MessageRole::User, MessageRole::User);
        assert_eq!(MessageRole::Assistant, MessageRole::Assistant);
        assert_ne!(MessageRole::User, MessageRole::Assistant);
        assert_ne!(MessageRole::System, MessageRole::Function);
    }

    // =========================================================================
    // FinishReason Tests
    // =========================================================================

    #[test]
    fn test_finish_reason_default_is_end_of_sequence() {
        let mut organ = LLMOrgan::default();
        let result = organ.generate("Test");
        assert_eq!(result.finish_reason, FinishReason::EndOfSequence);
    }

    #[test]
    fn test_finish_reason_equality() {
        assert_eq!(FinishReason::EndOfSequence, FinishReason::EndOfSequence);
        assert_ne!(FinishReason::EndOfSequence, FinishReason::MaxLength);
    }

    // =========================================================================
    // QueryType Tests
    // =========================================================================

    #[test]
    fn test_query_type_equality() {
        assert_eq!(QueryType::Generation, QueryType::Generation);
        assert_ne!(QueryType::Generation, QueryType::QA);
        assert_ne!(QueryType::Translation, QueryType::Code);
    }

    // =========================================================================
    // LLMGenerationResult Tests
    // =========================================================================

    #[test]
    fn test_generation_result_fields() {
        let mut organ = LLMOrgan::default();
        let result = organ.generate("Hello");

        assert!(!result.text.is_empty());
        assert!(result.confidence > 0.0);
        assert!(result.tokens_generated > 0);
        assert!(result.generation_time_ms >= 0.0);
        assert!(result.embedding.dim() > 0);
    }

    // =========================================================================
    // Statistics Tests
    // =========================================================================

    #[test]
    fn test_stats_accessor() {
        let mut organ = LLMOrgan::default();
        assert_eq!(organ.stats().queries_processed, 0);

        organ.generate("Test");
        assert_eq!(organ.stats().queries_processed, 1);
    }

    #[test]
    fn test_stats_accumulation() {
        let mut organ = LLMOrgan::default();
        organ.generate("First");
        organ.generate("Second");
        organ.generate("Third");

        assert_eq!(organ.stats().queries_processed, 3);
        assert!(organ.stats().tokens_generated > 0);
        assert!(organ.stats().avg_generation_time_ms >= 0.0);
    }

    // =========================================================================
    // Embedding Cache Tests
    // =========================================================================

    #[test]
    fn test_embedding_cache_hit() {
        let mut organ = LLMOrgan::default();

        // Generate twice with same content - second should hit cache
        organ.generate("Same text");
        let initial_hits = organ.stats().cache_hits;

        organ.generate("Same text");
        assert!(organ.stats().cache_hits > initial_hits);
    }

    // =========================================================================
    // ConsciousLlmOrgan Tests
    // =========================================================================

    #[test]
    fn test_conscious_llm_organ_creation() {
        let organ = ConsciousLlmOrgan::default();
        assert_eq!(organ.provider, LlmProvider::Local);
        assert!((organ.consciousness_level - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_conscious_llm_organ_with_provider() {
        let organ = ConsciousLlmOrgan::new(LLMOrganConfig::default(), LlmProvider::Ollama);
        assert_eq!(organ.provider, LlmProvider::Ollama);
    }

    #[test]
    fn test_set_consciousness_level() {
        let mut organ = ConsciousLlmOrgan::default();

        organ.set_consciousness_level(0.8);
        assert!((organ.consciousness_level - 0.8).abs() < 0.01);

        // Test clamping
        organ.set_consciousness_level(1.5);
        assert!((organ.consciousness_level - 1.0).abs() < 0.01);

        organ.set_consciousness_level(-0.5);
        assert!((organ.consciousness_level - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_conscious_query_adjusts_confidence() {
        let mut organ = ConsciousLlmOrgan::default();
        organ.set_consciousness_level(0.5);

        let query = LLMQuery {
            query_type: QueryType::Generation,
            content: "Test".to_string(),
            context: Vec::new(),
            system_prompt: None,
            params: None,
        };

        let result = organ.conscious_query(query);
        // Confidence should be adjusted by consciousness level
        assert!(result.confidence <= 0.85 * 0.5 + 0.01);
    }

    // =========================================================================
    // LlmProvider Tests
    // =========================================================================

    #[test]
    fn test_llm_provider_default() {
        let provider = LlmProvider::default();
        assert_eq!(provider, LlmProvider::Local);
    }

    #[test]
    fn test_llm_provider_variants() {
        assert_ne!(LlmProvider::Ollama, LlmProvider::OpenAI);
        assert_ne!(LlmProvider::Anthropic, LlmProvider::Local);
    }

    // =========================================================================
    // LLMQueryParams Tests
    // =========================================================================

    #[test]
    fn test_llm_query_params() {
        let params = LLMQueryParams {
            temperature: Some(0.3),
            max_length: Some(100),
            stop_sequences: vec!["STOP".to_string()],
        };

        assert_eq!(params.temperature, Some(0.3));
        assert_eq!(params.max_length, Some(100));
        assert_eq!(params.stop_sequences.len(), 1);
    }

    // =========================================================================
    // Type Alias Tests
    // =========================================================================

    #[test]
    fn test_type_aliases() {
        // These are just compile-time checks
        let _organ: LlmOrgan = LLMOrgan::default();
        let _config: LlmConfig = LLMOrganConfig::default();
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_empty_input() {
        let mut organ = LLMOrgan::default();
        let result = organ.generate("");
        // Should still produce a result, even if input is empty
        assert!(!result.text.is_empty());
    }

    #[test]
    fn test_very_long_input() {
        let mut organ = LLMOrgan::default();
        let long_input = "word ".repeat(1000);
        let result = organ.generate(&long_input);
        assert!(!result.text.is_empty());
    }

    #[test]
    fn test_special_characters_in_input() {
        let mut organ = LLMOrgan::default();
        let result = organ.generate("Hello! @#$%^&*() 你好 🎉");
        assert!(!result.text.is_empty());
    }

    // =========================================================================
    // Translation System Prompt Tests
    // =========================================================================

    #[test]
    fn test_translation_system_prompt_exists() {
        assert!(!TRANSLATION_SYSTEM_PROMPT.is_empty());
        assert!(TRANSLATION_SYSTEM_PROMPT.contains("TRANSLATION"));
        assert!(TRANSLATION_SYSTEM_PROMPT.contains("EPISTEMIC"));
    }
}

// ============================================================================
// TYPE ALIASES FOR INTEGRATION MODULE COMPATIBILITY
// ============================================================================
// The integration module uses different naming conventions.
// These aliases provide compatibility.

/// Type alias for camelCase naming convention
pub type LlmOrgan = LLMOrgan;

/// Type alias for camelCase config naming
pub type LlmConfig = LLMOrganConfig;

/// Type alias for query as request
pub type LlmRequest = LLMQuery;

/// Provider types for LLM backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlmProvider {
    /// Local Ollama instance
    Ollama,
    /// OpenAI API
    OpenAI,
    /// Anthropic Claude
    Anthropic,
    /// Custom/local model
    #[default]
    Local,
    /// Native CfC-HDC SSM language center (Broca's area)
    #[cfg(feature = "ssm_language")]
    Ssm,
    /// Liquid-Mamba: pre-trained SSM fused with HDC consciousness gating
    #[cfg(feature = "liquid-mamba")]
    LiquidMamba,
}

/// Consciousness-aware LLM wrapper
///
/// Wraps an LLMOrgan with consciousness integration capabilities.
#[derive(Debug)]
pub struct ConsciousLlmOrgan {
    /// Inner LLM organ
    pub inner: LLMOrgan,
    /// Provider type
    pub provider: LlmProvider,
    /// Current consciousness level (0.0-1.0)
    pub consciousness_level: f32,
}

impl ConsciousLlmOrgan {
    /// Create a new conscious LLM organ
    pub fn new(config: LLMOrganConfig, provider: LlmProvider) -> Self {
        Self {
            inner: LLMOrgan::new(config),
            provider,
            consciousness_level: 0.5,
        }
    }

    /// Process a query with consciousness awareness
    pub fn conscious_query(&mut self, query: LLMQuery) -> LLMGenerationResult {
        // Adjust response based on consciousness level
        let mut result = self.inner.query(query);
        result.confidence *= self.consciousness_level;
        result
    }

    /// Update consciousness level
    pub fn set_consciousness_level(&mut self, level: f32) {
        self.consciousness_level = level.clamp(0.0, 1.0);
    }
}

impl Default for ConsciousLlmOrgan {
    fn default() -> Self {
        Self::new(LLMOrganConfig::default(), LlmProvider::Local)
    }
}
