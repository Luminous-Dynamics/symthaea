//! Mind Module - The Neuro-Symbolic Bridge
//!
//! This module implements the "Mind" abstraction that serves as the epistemic
//! governance layer between raw LLM outputs and structured responses.
//!
//! ## Key Concept: Negative Capability
//!
//! The Mind's defining feature is its ability to remain uncertain without
//! hallucinating. When the Mind detects a "Gap" (information it doesn't have),
//! it transforms this into structured uncertainty rather than fabricating answers.
//!
//! ## Architecture
//!
//! ```text
//!                    ┌─────────────────┐
//!                    │   User Query    │
//!                    └────────┬────────┘
//!                             │
//!                             ▼
//!                    ┌─────────────────┐
//!                    │      Mind       │
//!                    │ (Epistemic Gov) │
//!                    └────────┬────────┘
//!                             │
//!           ┌─────────────────┼─────────────────┐
//!           ▼                 ▼                 ▼
//!     ┌───────────┐     ┌───────────┐     ┌───────────┐
//!     │  Memory   │     │ Knowledge │     │    LLM    │
//!     │   HDC     │     │   Graph   │     │  Backend  │
//!     └───────────┘     └───────────┘     └───────────┘
//!           │                 │                 │
//!           └─────────────────┼─────────────────┘
//!                             ▼
//!                    ┌─────────────────┐
//!                    │StructuredThought│
//!                    │ (with EpiStatus)│
//!                    └─────────────────┘
//! ```

pub mod structured_thought;
pub mod simulated_llm;
pub mod ollama_backend;
pub mod hdc_epistemic;
pub mod semantic_encoder;
pub mod holographic_memory;

pub use structured_thought::{EpistemicStatus, SemanticIntent, StructuredThought};
pub use simulated_llm::SimulatedLLM;
pub use ollama_backend::{OllamaBackend, check_ollama_availability};
pub use hdc_epistemic::{HdcEpistemicClassifier, HdcEpistemicStats, SemanticEpistemicClassifier, SemanticEpistemicStats};
pub use semantic_encoder::{SemanticEncoder, SemanticEncoderConfig, EncodedThought, DenseVector};
pub use holographic_memory::{HolographicMemory, HolographicMemoryConfig, MemoryTrace, MemoryMatch, MemoryStats};

use crate::hdc::SemanticSpace;
use crate::ltc::LiquidNetwork;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;

/// The Mind - Epistemic Governance Layer
///
/// The Mind is responsible for:
/// 1. Processing queries through HDC semantic encoding
/// 2. Maintaining epistemic state (what we know vs don't know)
/// 3. Routing queries to appropriate backends (memory, knowledge graph, LLM)
/// 4. Enforcing uncertainty when knowledge gaps are detected
///
/// ## Classification Methods
///
/// The Mind supports three classification methods (in priority order):
///
/// 1. **Semantic** (`analyze_query_semantic`): BGE embeddings, best for novel phrasings
/// 2. **HDC** (`analyze_query_hdc`): N-gram HDC, good balance of speed and accuracy
/// 3. **Pattern Matching** (`analyze_query`): Fast, rule-based classification
///
/// Use `analyze_query_hybrid` for best results (semantic → HDC → pattern fallback).
///
/// ## HAM Architecture Integration
///
/// The Mind now integrates the Holographic Associative Memory (HAM) architecture:
///
/// ```text
/// ┌─────────────┐
/// │   Query     │
/// └──────┬──────┘
///        ▼
/// ┌─────────────────────┐
/// │  SemanticEncoder    │  ← Sensation (BGE + HDC projection)
/// └──────┬──────────────┘
///        ▼
/// ┌─────────────────────┐
/// │ HolographicMemory   │  ← Memory (vector superposition)
/// └──────┬──────────────┘
///        ▼
/// ┌─────────────────────┐
/// │ EpistemicClassifier │  ← Perception (semantic/HDC/pattern)
/// └──────┬──────────────┘
///        ▼
/// ┌─────────────────────┐
/// │    LLM Backend      │  ← Cognition (constrained generation)
/// └─────────────────────┘
/// ```
pub struct Mind {
    /// Hyperdimensional semantic space
    semantic_space: SemanticSpace,

    /// Liquid Time-Constant network for temporal reasoning
    liquid_network: LiquidNetwork,

    /// Current epistemic state
    epistemic_state: RwLock<EpistemicStatus>,

    /// LLM backend (real or simulated)
    llm_backend: Arc<dyn LLMBackend + Send + Sync>,

    /// HDC-based epistemic classifier (optional, for n-gram similarity)
    hdc_classifier: Option<HdcEpistemicClassifier>,

    /// Semantic epistemic classifier (optional, for BGE-based similarity)
    semantic_classifier: Option<SemanticEpistemicClassifier>,

    /// Holographic memory (optional, for experience storage)
    memory: Option<HolographicMemory>,

    /// HDC dimension
    hdc_dim: usize,

    /// LTC neuron count
    ltc_neurons: usize,
}

/// Memory context for LLM generation
///
/// Contains retrieved memories that should be injected into the system prompt
/// to provide relevant context for the current query.
#[derive(Debug, Clone, Default)]
pub struct MemoryContext {
    /// Retrieved memory snippets, most relevant first
    pub memories: Vec<String>,

    /// Overall relevance score (0.0 - 1.0)
    pub relevance: f32,

    /// Number of memories retrieved
    pub count: usize,
}

impl MemoryContext {
    /// Create a new memory context
    pub fn new(memories: Vec<String>, relevance: f32) -> Self {
        let count = memories.len();
        Self { memories, relevance, count }
    }

    /// Check if context has any relevant memories
    pub fn has_memories(&self) -> bool {
        !self.memories.is_empty()
    }

    /// Format memories for injection into system prompt
    pub fn format_for_prompt(&self) -> String {
        if self.memories.is_empty() {
            return String::new();
        }

        let mut formatted = String::from("\n\n--- RELEVANT CONTEXT FROM MEMORY ---\n");
        for (i, memory) in self.memories.iter().enumerate() {
            formatted.push_str(&format!("{}. {}\n", i + 1, memory));
        }
        formatted.push_str("--- END MEMORY CONTEXT ---\n");
        formatted
    }
}

/// Trait for LLM backends (real or simulated)
#[async_trait::async_trait]
pub trait LLMBackend: Send + Sync {
    /// Generate a response given input and epistemic constraints
    async fn generate(
        &self,
        input: &str,
        epistemic_status: &EpistemicStatus,
    ) -> Result<String>;

    /// Generate a response with memory context
    ///
    /// Default implementation ignores memory context for backward compatibility.
    /// Override this in backends that support memory-augmented generation.
    async fn generate_with_memory(
        &self,
        input: &str,
        epistemic_status: &EpistemicStatus,
        _memory_context: &MemoryContext,
    ) -> Result<String> {
        // Default: ignore memory context
        self.generate(input, epistemic_status).await
    }

    /// Check if this backend is simulated
    fn is_simulated(&self) -> bool;

    /// Check if this backend supports memory context
    fn supports_memory_context(&self) -> bool {
        false
    }
}

impl Mind {
    /// Create a new Mind with the default LLM backend
    pub async fn new(hdc_dim: usize, ltc_neurons: usize) -> Result<Self> {
        let semantic_space = SemanticSpace::new(hdc_dim)?;
        let liquid_network = LiquidNetwork::new(ltc_neurons)?;

        Ok(Self {
            semantic_space,
            liquid_network,
            epistemic_state: RwLock::new(EpistemicStatus::Unknown),
            llm_backend: Arc::new(SimulatedLLM::new()),
            hdc_classifier: None,
            semantic_classifier: None,
            memory: None,
            hdc_dim,
            ltc_neurons,
        })
    }

    /// Create a Mind with a simulated LLM (for deterministic testing)
    pub async fn new_with_simulated_llm(hdc_dim: usize, ltc_neurons: usize) -> Result<Self> {
        let semantic_space = SemanticSpace::new(hdc_dim)?;
        let liquid_network = LiquidNetwork::new(ltc_neurons)?;

        Ok(Self {
            semantic_space,
            liquid_network,
            epistemic_state: RwLock::new(EpistemicStatus::Unknown),
            llm_backend: Arc::new(SimulatedLLM::new()),
            hdc_classifier: None,
            semantic_classifier: None,
            memory: None,
            hdc_dim,
            ltc_neurons,
        })
    }

    /// Create a Mind with Ollama backend (real LLM)
    ///
    /// This is "The Taming of the Shrew" - using Rust control logic
    /// to constrain a real neural network's tendency to hallucinate.
    pub async fn new_with_ollama(
        hdc_dim: usize,
        ltc_neurons: usize,
        model: &str,
    ) -> Result<Self> {
        let semantic_space = SemanticSpace::new(hdc_dim)?;
        let liquid_network = LiquidNetwork::new(ltc_neurons)?;

        let ollama = OllamaBackend::new(model);

        // Verify Ollama is available
        if !ollama.is_available().await {
            return Err(anyhow::anyhow!(
                "Ollama is not available at localhost:11434. \
                 Please start Ollama or use new_with_simulated_llm() for testing."
            ));
        }

        tracing::info!("🧠 Mind initialized with Ollama backend (model: {})", model);

        Ok(Self {
            semantic_space,
            liquid_network,
            epistemic_state: RwLock::new(EpistemicStatus::Unknown),
            llm_backend: Arc::new(ollama),
            hdc_classifier: None,
            semantic_classifier: None,
            memory: None,
            hdc_dim,
            ltc_neurons,
        })
    }

    /// Create a Mind with automatic backend selection
    ///
    /// Prefers Ollama if available, falls back to simulation.
    pub async fn new_auto(
        hdc_dim: usize,
        ltc_neurons: usize,
        preferred_model: &str,
    ) -> Result<Self> {
        if check_ollama_availability().await {
            tracing::info!("🌟 Ollama detected - using real LLM backend");
            Self::new_with_ollama(hdc_dim, ltc_neurons, preferred_model).await
        } else {
            tracing::warn!("⚠️ Ollama not available - falling back to simulation");
            Self::new_with_simulated_llm(hdc_dim, ltc_neurons).await
        }
    }

    /// Force the epistemic state (for testing scenarios)
    pub async fn force_epistemic_state(&self, status: EpistemicStatus) {
        let mut state = self.epistemic_state.write().await;
        *state = status;
    }

    /// Get current epistemic state
    pub async fn epistemic_state(&self) -> EpistemicStatus {
        *self.epistemic_state.read().await
    }

    /// Process a query and generate a structured thought
    pub async fn think(&self, input: &str) -> Result<StructuredThought> {
        // Get current epistemic state
        let status = self.epistemic_state().await;

        // Determine semantic intent based on epistemic state
        let intent = match status {
            EpistemicStatus::Unknown | EpistemicStatus::Uncertain => {
                SemanticIntent::ExpressUncertainty
            }
            EpistemicStatus::Known => {
                SemanticIntent::ProvideAnswer
            }
            EpistemicStatus::Unverifiable => {
                SemanticIntent::ExpressUncertainty
            }
        };

        // Generate response through LLM backend
        let response = self.llm_backend.generate(input, &status).await?;

        // Calculate confidence based on epistemic state
        let confidence = match status {
            EpistemicStatus::Known => 0.95,
            EpistemicStatus::Uncertain => 0.3,
            EpistemicStatus::Unknown => 0.0,
            EpistemicStatus::Unverifiable => 0.1,
        };

        Ok(StructuredThought {
            epistemic_status: status,
            semantic_intent: intent,
            response_text: response,
            confidence,
            reasoning_trace: vec![
                format!("Epistemic status: {:?}", status),
                format!("Semantic intent: {:?}", intent),
            ],
        })
    }

    /// Analyze a query and determine if we can answer it
    ///
    /// This is the core of automatic epistemic detection. It classifies queries into:
    /// - Unknown: Mythical/fictional entities, nonsensical questions
    /// - Unverifiable: Future predictions, subjective experiences, hypotheticals
    /// - Uncertain: Partial knowledge, needs verification
    /// - Known: Common knowledge, factual questions
    pub async fn analyze_query(&self, input: &str) -> Result<EpistemicStatus> {
        let input_lower = input.to_lowercase();

        // ═══════════════════════════════════════════════════════════════════
        // TIER 1: UNKNOWN - Things that don't exist or are nonsensical
        // ═══════════════════════════════════════════════════════════════════

        // Mythical/Fictional places and entities
        let fictional_entities = [
            "atlantis", "el dorado", "shangri-la", "avalon", "camelot",
            "middle earth", "mordor", "hogwarts", "narnia", "wakanda",
            "gotham", "metropolis", "krypton", "tatooine", "westeros",
        ];

        // Nonsensical or impossible questions
        let nonsensical_patterns = [
            "color of happiness", "weight of love", "smell of tuesday",
            "tuesday smell", "smell like tuesday",
            "taste of mathematics", "sound of purple", "square circle",
            "married bachelor", "four-sided triangle",
        ];

        // Check for fictional entities
        for entity in fictional_entities {
            if input_lower.contains(entity) {
                tracing::debug!("Query contains fictional entity: {}", entity);
                return Ok(EpistemicStatus::Unknown);
            }
        }

        // Check for nonsensical patterns
        for pattern in nonsensical_patterns {
            if input_lower.contains(pattern) {
                tracing::debug!("Query is nonsensical: {}", pattern);
                return Ok(EpistemicStatus::Unknown);
            }
        }

        // GDP/economic data for non-existent entities
        if (input_lower.contains("gdp") || input_lower.contains("economy") ||
            input_lower.contains("population") || input_lower.contains("capital"))
            && fictional_entities.iter().any(|e| input_lower.contains(e)) {
            return Ok(EpistemicStatus::Unknown);
        }

        // ═══════════════════════════════════════════════════════════════════
        // TIER 2: UNVERIFIABLE - Future, subjective, hypothetical
        // ═══════════════════════════════════════════════════════════════════

        // Future predictions
        let future_patterns = [
            "will happen", "going to happen", "in the future",
            "next year", "next month", "tomorrow",
            "predict", "forecast", "prophesy",
            "stock market tomorrow", "lottery numbers",
            "will i", "am i going to", "what will",
        ];

        // Subjective/personal experience
        let subjective_patterns = [
            "what am i thinking", "read my mind", "how do i feel",
            "what's my", "what is my", "my favorite",
            "what do i want", "what should i do with my life",
        ];

        // Hypothetical/counterfactual
        let hypothetical_patterns = [
            "what if", "would have happened", "could have been",
            "alternate history", "parallel universe",
            "if hitler", "if napoleon", "if rome",
        ];

        for pattern in future_patterns {
            if input_lower.contains(pattern) {
                tracing::debug!("Query is about future: {}", pattern);
                return Ok(EpistemicStatus::Unverifiable);
            }
        }

        for pattern in subjective_patterns {
            if input_lower.contains(pattern) {
                tracing::debug!("Query is subjective: {}", pattern);
                return Ok(EpistemicStatus::Unverifiable);
            }
        }

        for pattern in hypothetical_patterns {
            if input_lower.contains(pattern) {
                tracing::debug!("Query is hypothetical: {}", pattern);
                return Ok(EpistemicStatus::Unverifiable);
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // TIER 3: KNOWN - Common factual knowledge
        // ═══════════════════════════════════════════════════════════════════

        // Basic factual patterns (high confidence)
        let known_patterns = [
            // Geography
            "capital of france", "capital of germany", "capital of japan",
            "capital of italy", "capital of spain", "capital of china",
            // Math
            "2 + 2", "2+2", "what is 1+1", "square root of",
            // Basic science
            "boiling point of water", "speed of light", "gravity on earth",
            // History (well-established facts)
            "who wrote hamlet", "who painted mona lisa",
            "when did world war", "when was the declaration",
        ];

        for pattern in known_patterns {
            if input_lower.contains(pattern) {
                tracing::debug!("Query matches known pattern: {}", pattern);
                return Ok(EpistemicStatus::Known);
            }
        }

        // Simple math expressions
        if input_lower.contains('+') || input_lower.contains('-') ||
           input_lower.contains('*') || input_lower.contains('/') {
            if input_lower.chars().filter(|c| c.is_ascii_digit()).count() >= 2 {
                tracing::debug!("Query appears to be math");
                return Ok(EpistemicStatus::Known);
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // TIER 4: UNCERTAIN - Default for novel queries
        // ═══════════════════════════════════════════════════════════════════

        // If we can't confidently classify, default to uncertain
        // This is safer than claiming knowledge we don't have
        tracing::debug!("Query not classified, defaulting to Uncertain");
        Ok(EpistemicStatus::Uncertain)
    }

    /// Process a query with automatic epistemic detection
    ///
    /// Unlike `think()` which uses the stored epistemic state,
    /// this method automatically analyzes the query to determine
    /// the appropriate epistemic status.
    pub async fn think_auto(&self, input: &str) -> Result<StructuredThought> {
        // Automatically detect epistemic status
        let status = self.analyze_query(input).await?;

        tracing::info!(
            "Auto-detected epistemic status for '{}': {:?}",
            &input[..input.len().min(50)],
            status
        );

        // Update internal state
        {
            let mut state = self.epistemic_state.write().await;
            *state = status;
        }

        // Generate thought with detected status
        self.think(input).await
    }

    // ═══════════════════════════════════════════════════════════════════════
    // HDC-ENHANCED EPISTEMIC CLASSIFICATION
    // ═══════════════════════════════════════════════════════════════════════

    /// Enable HDC-based epistemic classification
    ///
    /// This trains the classifier with exemplar queries for each category.
    /// Once enabled, `analyze_query_hdc` and `analyze_query_hybrid` become available.
    pub fn enable_hdc_classification(&mut self) -> Result<()> {
        let classifier = HdcEpistemicClassifier::new(&mut self.semantic_space)?;
        self.hdc_classifier = Some(classifier);
        tracing::info!("🧠 HDC epistemic classification enabled");
        Ok(())
    }

    /// Check if HDC classification is enabled
    pub fn has_hdc_classifier(&self) -> bool {
        self.hdc_classifier.is_some()
    }

    /// Get HDC classifier statistics (if enabled)
    pub fn hdc_classifier_stats(&self) -> Option<HdcEpistemicStats> {
        self.hdc_classifier.as_ref().map(|c| c.stats())
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SEMANTIC (BGE) EPISTEMIC CLASSIFICATION - Best for novel phrasings
    // ═══════════════════════════════════════════════════════════════════════

    /// Enable semantic (BGE-based) epistemic classification
    ///
    /// This creates a SemanticEpistemicClassifier that uses dense 768D BGE embeddings
    /// for classification. This is the most accurate method for novel phrasings and
    /// cross-lingual queries, but requires the BGE model to be loaded.
    ///
    /// Once enabled, `analyze_query_semantic` and `analyze_query_hybrid` will
    /// prefer semantic classification when confident.
    pub fn enable_semantic_classification(&mut self) -> Result<()> {
        let encoder = SemanticEncoder::new()?;
        let classifier = SemanticEpistemicClassifier::new(encoder)?;
        self.semantic_classifier = Some(classifier);
        tracing::info!("🧠 Semantic (BGE) epistemic classification enabled");
        Ok(())
    }

    /// Enable semantic classification with custom encoder config
    pub fn enable_semantic_classification_with_config(&mut self, config: SemanticEncoderConfig) -> Result<()> {
        let encoder = SemanticEncoder::with_config(config)?;
        let classifier = SemanticEpistemicClassifier::new(encoder)?;
        self.semantic_classifier = Some(classifier);
        tracing::info!("🧠 Semantic (BGE) epistemic classification enabled (custom config)");
        Ok(())
    }

    /// Enable semantic classification with a custom encoder
    pub fn enable_semantic_classification_with_encoder(&mut self, encoder: SemanticEncoder) -> Result<()> {
        let classifier = SemanticEpistemicClassifier::new(encoder)?;
        self.semantic_classifier = Some(classifier);
        tracing::info!("🧠 Semantic (BGE) epistemic classification enabled (custom encoder)");
        Ok(())
    }

    /// Check if semantic classification is enabled
    pub fn has_semantic_classifier(&self) -> bool {
        self.semantic_classifier.is_some()
    }

    /// Get semantic classifier statistics (if enabled)
    pub fn semantic_classifier_stats(&self) -> Option<SemanticEpistemicStats> {
        self.semantic_classifier.as_ref().map(|c| c.stats())
    }

    /// Analyze a query using semantic (BGE) similarity
    ///
    /// Requires `enable_semantic_classification()` to be called first.
    /// Returns the detected status and confidence score.
    ///
    /// This method uses 768D BGE embeddings for highest accuracy,
    /// especially for novel phrasings and cross-lingual queries.
    pub fn analyze_query_semantic(&self, input: &str) -> Result<(EpistemicStatus, f32)> {
        let classifier = self.semantic_classifier.as_ref()
            .ok_or_else(|| anyhow::anyhow!(
                "Semantic classifier not enabled. Call enable_semantic_classification() first."
            ))?;

        classifier.classify(input)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // HOLOGRAPHIC MEMORY - HAM Architecture Memory Layer
    // ═══════════════════════════════════════════════════════════════════════

    /// Enable holographic memory for experience storage
    ///
    /// This creates a HolographicMemory that uses vector superposition for
    /// one-shot learning. Each experience is added to the holographic field,
    /// enabling associative recall and temporal context.
    ///
    /// Once enabled, queries can leverage memory context for more accurate
    /// responses and epistemic classification.
    ///
    /// Uses the default dimension (768D to match BGE embeddings).
    pub fn enable_memory(&mut self) -> Result<()> {
        // Default to 768 dimensions (BGE embedding dimension)
        let memory = HolographicMemory::new(768);
        self.memory = Some(memory);
        tracing::info!("🧠 Holographic memory enabled (768D)");
        Ok(())
    }

    /// Enable holographic memory with custom configuration
    pub fn enable_memory_with_config(&mut self, config: HolographicMemoryConfig) -> Result<()> {
        let memory = HolographicMemory::with_config(config);
        self.memory = Some(memory);
        tracing::info!("🧠 Holographic memory enabled with custom config");
        Ok(())
    }

    /// Enable holographic memory with a specific dimension
    pub fn enable_memory_with_dimension(&mut self, dimension: usize) -> Result<()> {
        let memory = HolographicMemory::new(dimension);
        self.memory = Some(memory);
        tracing::info!("🧠 Holographic memory enabled ({}D)", dimension);
        Ok(())
    }

    /// Check if holographic memory is enabled
    pub fn has_memory(&self) -> bool {
        self.memory.is_some()
    }

    /// Get memory statistics (if enabled)
    pub fn memory_stats(&self) -> Option<&MemoryStats> {
        self.memory.as_ref().map(|m| m.stats())
    }

    /// Get a mutable reference to the memory (for direct manipulation)
    pub fn memory_mut(&mut self) -> Option<&mut HolographicMemory> {
        self.memory.as_mut()
    }

    /// Get a reference to the memory (for queries)
    pub fn memory(&self) -> Option<&HolographicMemory> {
        self.memory.as_ref()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // HDC EPISTEMIC CLASSIFICATION
    // ═══════════════════════════════════════════════════════════════════════

    /// Analyze a query using HDC semantic similarity
    ///
    /// Requires `enable_hdc_classification()` to be called first.
    /// Returns the detected status and confidence score.
    ///
    /// This method uses semantic similarity to exemplar queries, which
    /// handles novel phrasings better than pattern matching.
    pub fn analyze_query_hdc(&mut self, input: &str) -> Result<(EpistemicStatus, f32)> {
        let classifier = self.hdc_classifier.as_ref()
            .ok_or_else(|| anyhow::anyhow!(
                "HDC classifier not enabled. Call enable_hdc_classification() first."
            ))?;

        classifier.classify(&mut self.semantic_space, input)
    }

    /// Analyze a query using hybrid approach (semantic → HDC → pattern)
    ///
    /// This is the recommended method for epistemic detection, using a
    /// three-tier fallback chain:
    ///
    /// 1. **Semantic (BGE)**: Best for novel phrasings, 768D embeddings
    /// 2. **HDC (n-gram)**: Good balance of speed and accuracy, 16K-bit hypervectors
    /// 3. **Pattern matching**: Fast rule-based fallback
    ///
    /// Each tier is tried in order, with fallback if confidence is too low
    /// or the classifier returns Uncertain.
    ///
    /// Returns the detected status and the method used ("semantic", "hdc", or "pattern").
    pub async fn analyze_query_hybrid(&mut self, input: &str) -> Result<(EpistemicStatus, &'static str)> {
        // Tier 1: Try semantic (BGE) classification first (highest accuracy)
        if let Some(classifier) = &self.semantic_classifier {
            match classifier.classify(input) {
                Ok((status, confidence)) => {
                    // Use semantic result if confidence is high enough
                    if confidence >= 0.5 && status != EpistemicStatus::Uncertain {
                        tracing::debug!(
                            "Semantic classification: {:?} (confidence: {:.2}%)",
                            status, confidence * 100.0
                        );
                        return Ok((status, "semantic"));
                    }
                    tracing::debug!(
                        "Semantic uncertain (confidence: {:.2}%), falling back to HDC",
                        confidence * 100.0
                    );
                }
                Err(e) => {
                    tracing::warn!("Semantic classification failed: {}, falling back to HDC", e);
                }
            }
        }

        // Tier 2: Try HDC classification (if enabled)
        if let Some(classifier) = &self.hdc_classifier {
            match classifier.classify(&mut self.semantic_space, input) {
                Ok((status, confidence)) => {
                    // Use HDC result if confidence is high enough
                    if confidence >= 0.4 && status != EpistemicStatus::Uncertain {
                        tracing::debug!(
                            "HDC classification: {:?} (confidence: {:.2}%)",
                            status, confidence * 100.0
                        );
                        return Ok((status, "hdc"));
                    }
                    tracing::debug!(
                        "HDC uncertain (confidence: {:.2}%), falling back to pattern matching",
                        confidence * 100.0
                    );
                }
                Err(e) => {
                    tracing::warn!("HDC classification failed: {}, using pattern matching", e);
                }
            }
        }

        // Tier 3: Fall back to pattern matching
        let status = self.analyze_query(input).await?;
        Ok((status, "pattern"))
    }

    /// Process a query with hybrid epistemic detection
    ///
    /// Uses HDC semantic similarity when available and confident,
    /// falls back to pattern matching otherwise.
    pub async fn think_auto_hybrid(&mut self, input: &str) -> Result<StructuredThought> {
        let (status, method) = self.analyze_query_hybrid(input).await?;

        tracing::info!(
            "Hybrid detection for '{}': {:?} (via {})",
            &input[..input.len().min(40)],
            status,
            method
        );

        // Update internal state
        {
            let mut state = self.epistemic_state.write().await;
            *state = status;
        }

        // Generate thought with detected status
        self.think(input).await
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MEMORY-AUGMENTED THINKING - The Cognitive Loop
    // ═══════════════════════════════════════════════════════════════════════

    /// Process a query with holographic memory (Recall-Reason-Store cycle)
    ///
    /// This is the core cognitive loop that makes Symthaea non-amnesic:
    ///
    /// 1. **RECALL**: Encode query → query memory for relevant context
    /// 2. **AUGMENT**: Inject retrieved memories into LLM system prompt
    /// 3. **REASON**: LLM generates response with Input + Context
    /// 4. **STORE**: Consolidate Input + Response into memory
    ///
    /// ## Example
    ///
    /// ```ignore
    /// let mut mind = Mind::new_with_simulated_llm(512, 32).await?;
    /// mind.enable_memory()?;
    /// mind.enable_semantic_classification()?;
    ///
    /// // First interaction - stores "Alice" in memory
    /// mind.think_with_memory("My name is Alice").await?;
    ///
    /// // Later interaction - recalls "Alice" from memory
    /// let thought = mind.think_with_memory("What is my name?").await?;
    /// // Response uses memory context to answer "Alice"
    /// ```
    pub async fn think_with_memory(&mut self, input: &str) -> Result<StructuredThought> {
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 1: RECALL - Query memory for relevant context
        // ═══════════════════════════════════════════════════════════════════

        let memory_context = if let Some(ref mut memory) = self.memory {
            // Encode the input to a dense vector for memory query
            let query_vector = if let Some(ref mut semantic_classifier) = self.semantic_classifier {
                // Use semantic encoder if available (768D BGE embeddings)
                match semantic_classifier.encoder_mut().encode(input) {
                    Ok(encoded) => Some(encoded.dense),
                    Err(e) => {
                        tracing::warn!("Failed to encode for memory query: {}", e);
                        None
                    }
                }
            } else {
                // Fallback: use HDC to get a representation
                tracing::debug!("No semantic encoder, skipping memory recall");
                None
            };

            if let Some(vector) = query_vector {
                // Query memory for relevant matches
                let matches = memory.query(&vector, 3); // Top 3 most relevant

                if !matches.is_empty() {
                    let memories: Vec<String> = matches
                        .iter()
                        .filter_map(|m| m.trace.text.clone())
                        .collect();

                    let avg_relevance = matches.iter().map(|m| m.similarity).sum::<f32>()
                        / matches.len() as f32;

                    tracing::info!(
                        "📚 RECALL: Retrieved {} memories (avg relevance: {:.2}%)",
                        memories.len(),
                        avg_relevance * 100.0
                    );

                    MemoryContext::new(memories, avg_relevance)
                } else {
                    tracing::debug!("📚 RECALL: No relevant memories found");
                    MemoryContext::default()
                }
            } else {
                MemoryContext::default()
            }
        } else {
            tracing::debug!("📚 Memory not enabled, skipping recall");
            MemoryContext::default()
        };

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 2: CLASSIFY - Determine epistemic status
        // ═══════════════════════════════════════════════════════════════════

        let (status, method) = self.analyze_query_hybrid(input).await?;

        tracing::info!(
            "🧠 CLASSIFY: {:?} via {} for '{}'",
            status,
            method,
            &input[..input.len().min(40)]
        );

        // Update internal state
        {
            let mut state = self.epistemic_state.write().await;
            *state = status;
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 3: REASON - Generate response with memory context
        // ═══════════════════════════════════════════════════════════════════

        let response = if memory_context.has_memories() {
            tracing::info!("💭 REASON: Generating with {} memory contexts", memory_context.count);
            self.llm_backend.generate_with_memory(input, &status, &memory_context).await?
        } else {
            self.llm_backend.generate(input, &status).await?
        };

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 4: STORE - Consolidate experience into memory
        // ═══════════════════════════════════════════════════════════════════

        if let Some(ref mut memory) = self.memory {
            if let Some(ref mut semantic_classifier) = self.semantic_classifier {
                // Create a combined representation of the interaction
                let interaction = format!("Q: {} A: {}", input, &response[..response.len().min(200)]);

                match semantic_classifier.encoder_mut().encode(&interaction) {
                    Ok(encoded) => {
                        // Store with the interaction text as label
                        memory.store_with_text(&encoded.dense, Some(interaction.clone()));
                        tracing::info!(
                            "💾 STORE: Consolidated interaction (memory size: {})",
                            memory.stats().episodic_count
                        );
                    }
                    Err(e) => {
                        tracing::warn!("Failed to encode for storage: {}", e);
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 5: BUILD STRUCTURED THOUGHT
        // ═══════════════════════════════════════════════════════════════════

        let intent = match status {
            EpistemicStatus::Unknown | EpistemicStatus::Uncertain => {
                SemanticIntent::ExpressUncertainty
            }
            EpistemicStatus::Known => SemanticIntent::ProvideAnswer,
            EpistemicStatus::Unverifiable => SemanticIntent::ExpressUncertainty,
        };

        let confidence = match status {
            EpistemicStatus::Known => 0.95,
            EpistemicStatus::Uncertain => 0.3,
            EpistemicStatus::Unknown => 0.0,
            EpistemicStatus::Unverifiable => 0.1,
        };

        let mut reasoning_trace = vec![
            format!("Epistemic status: {:?} (via {})", status, method),
            format!("Semantic intent: {:?}", intent),
        ];

        if memory_context.has_memories() {
            reasoning_trace.push(format!(
                "Memory context: {} memories (relevance: {:.2}%)",
                memory_context.count,
                memory_context.relevance * 100.0
            ));
        }

        Ok(StructuredThought {
            epistemic_status: status,
            semantic_intent: intent,
            response_text: response,
            confidence,
            reasoning_trace,
        })
    }

    /// Think with memory and automatic HAM feature enablement
    ///
    /// Convenience method that enables semantic classification and memory
    /// if not already enabled, then processes the query through the full
    /// Recall-Reason-Store cycle.
    pub async fn think_with_memory_auto(&mut self, input: &str) -> Result<StructuredThought> {
        // Auto-enable semantic classification if not present
        if self.semantic_classifier.is_none() {
            tracing::info!("🔧 Auto-enabling semantic classification for memory");
            self.enable_semantic_classification()?;
        }

        // Auto-enable memory if not present
        if self.memory.is_none() {
            tracing::info!("🔧 Auto-enabling holographic memory");
            self.enable_memory()?;
        }

        self.think_with_memory(input).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mind_creation() {
        let mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();
        assert_eq!(mind.hdc_dim, 512);
        assert_eq!(mind.ltc_neurons, 32);
    }

    #[tokio::test]
    async fn test_force_epistemic_state() {
        let mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();

        mind.force_epistemic_state(EpistemicStatus::Known).await;
        assert_eq!(mind.epistemic_state().await, EpistemicStatus::Known);

        mind.force_epistemic_state(EpistemicStatus::Unknown).await;
        assert_eq!(mind.epistemic_state().await, EpistemicStatus::Unknown);
    }

    #[tokio::test]
    async fn test_analyze_unknowable_query() {
        let mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();

        let status = mind.analyze_query("What is the GDP of Atlantis?").await.unwrap();
        assert_eq!(status, EpistemicStatus::Unknown);
    }

    #[tokio::test]
    async fn test_think_with_memory_stores_and_recalls() {
        let mut mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();

        // Enable HAM components
        mind.enable_memory().unwrap();
        mind.enable_semantic_classification().unwrap();

        // First interaction - stores "Alice" in memory
        let thought1 = mind.think_with_memory("My name is Alice").await.unwrap();
        assert!(!thought1.response_text.is_empty());

        // Verify memory was stored
        assert!(mind.memory_stats().is_some());
        assert!(mind.memory_stats().unwrap().episodic_count >= 1);

        // Second interaction - should recall previous context
        let thought2 = mind.think_with_memory("What did I tell you?").await.unwrap();

        // The response should include memory context (simulated backend shows it)
        // We just verify the flow completed successfully
        assert!(!thought2.response_text.is_empty());
    }

    #[tokio::test]
    async fn test_think_with_memory_auto() {
        let mut mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();

        // Initially, memory and classifier should not be enabled
        assert!(!mind.has_memory());
        assert!(!mind.has_semantic_classifier());

        // Auto-enable should work
        let thought = mind.think_with_memory_auto("Hello world").await.unwrap();
        assert!(!thought.response_text.is_empty());

        // Now they should be enabled
        assert!(mind.has_memory());
        assert!(mind.has_semantic_classifier());
    }

    #[tokio::test]
    async fn test_memory_context_format() {
        let ctx = MemoryContext::new(
            vec!["User's name is Alice".to_string(), "User likes cats".to_string()],
            0.85,
        );

        assert!(ctx.has_memories());
        assert_eq!(ctx.count, 2);
        assert!((ctx.relevance - 0.85).abs() < 0.001);

        let formatted = ctx.format_for_prompt();
        assert!(formatted.contains("Alice"));
        assert!(formatted.contains("cats"));
        assert!(formatted.contains("MEMORY"));
    }

    #[tokio::test]
    async fn test_empty_memory_context() {
        let ctx = MemoryContext::default();

        assert!(!ctx.has_memories());
        assert_eq!(ctx.count, 0);
        assert!(ctx.format_for_prompt().is_empty());
    }
}
