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

pub use structured_thought::{EpistemicStatus, SemanticIntent, StructuredThought};
pub use simulated_llm::SimulatedLLM;
pub use ollama_backend::{OllamaBackend, check_ollama_availability};
pub use hdc_epistemic::{HdcEpistemicClassifier, HdcEpistemicStats};

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
/// The Mind supports two classification methods:
///
/// 1. **Pattern Matching** (`analyze_query`): Fast, rule-based classification
/// 2. **HDC Semantic** (`analyze_query_hdc`): Similarity-based, handles novel phrasings
///
/// Use `analyze_query_hybrid` for best results (HDC with pattern matching fallback).
pub struct Mind {
    /// Hyperdimensional semantic space
    semantic_space: SemanticSpace,

    /// Liquid Time-Constant network for temporal reasoning
    liquid_network: LiquidNetwork,

    /// Current epistemic state
    epistemic_state: RwLock<EpistemicStatus>,

    /// LLM backend (real or simulated)
    llm_backend: Arc<dyn LLMBackend + Send + Sync>,

    /// HDC-based epistemic classifier (optional, for semantic similarity)
    hdc_classifier: Option<HdcEpistemicClassifier>,

    /// HDC dimension
    hdc_dim: usize,

    /// LTC neuron count
    ltc_neurons: usize,
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

    /// Check if this backend is simulated
    fn is_simulated(&self) -> bool;
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

    /// Analyze a query using hybrid approach (HDC + pattern matching)
    ///
    /// This is the recommended method for epistemic detection:
    /// 1. First tries HDC classification (if enabled and confident)
    /// 2. Falls back to pattern matching if HDC is uncertain
    ///
    /// Returns the detected status and the method used ("hdc" or "pattern").
    pub async fn analyze_query_hybrid(&mut self, input: &str) -> Result<(EpistemicStatus, &'static str)> {
        // Try HDC classification first (if enabled)
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

        // Fall back to pattern matching
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
}
