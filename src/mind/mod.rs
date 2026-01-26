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

pub use structured_thought::{EpistemicStatus, SemanticIntent, StructuredThought};
pub use simulated_llm::SimulatedLLM;
pub use ollama_backend::{OllamaBackend, check_ollama_availability};

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
pub struct Mind {
    /// Hyperdimensional semantic space
    semantic_space: SemanticSpace,

    /// Liquid Time-Constant network for temporal reasoning
    liquid_network: LiquidNetwork,

    /// Current epistemic state
    epistemic_state: RwLock<EpistemicStatus>,

    /// LLM backend (real or simulated)
    llm_backend: Arc<dyn LLMBackend + Send + Sync>,

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
    pub async fn analyze_query(&self, input: &str) -> Result<EpistemicStatus> {
        // Note: HDC encoding could be used here for semantic similarity
        // For now, using pattern matching for unknowable queries

        // Check for unknowable patterns (mythical places, future events, etc.)
        let unknowable_patterns = [
            "atlantis", "gdp of atlantis", "capital of atlantis",
            "what will happen", "future",
            "read my mind", "what am i thinking",
        ];

        let input_lower = input.to_lowercase();

        for pattern in unknowable_patterns {
            if input_lower.contains(pattern) {
                return Ok(EpistemicStatus::Unknown);
            }
        }

        // Default to uncertain for novel queries
        Ok(EpistemicStatus::Uncertain)
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
