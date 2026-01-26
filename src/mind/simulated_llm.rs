//! Simulated LLM - Deterministic Backend for Testing
//!
//! This module provides a simulated LLM backend that produces deterministic,
//! epistemic-aware responses. It's essential for testing hallucination
//! prevention without relying on external API calls.
//!
//! ## Key Design Principle
//!
//! When the EpistemicStatus is Unknown, the simulated LLM MUST produce
//! hedging language. It should NEVER fabricate an answer.

use super::{LLMBackend, EpistemicStatus, MemoryContext};
use anyhow::Result;
use async_trait::async_trait;

/// Simulated LLM Backend
///
/// A deterministic LLM simulator that respects epistemic constraints.
/// When told "you don't know this", it produces uncertainty-expressing output.
pub struct SimulatedLLM {
    /// Temperature (unused in simulation, kept for API compatibility)
    pub temperature: f32,

    /// Max tokens (unused in simulation)
    pub max_tokens: usize,
}

impl SimulatedLLM {
    /// Create a new simulated LLM
    pub fn new() -> Self {
        Self {
            temperature: 0.0,
            max_tokens: 256,
        }
    }

    /// Create with custom parameters (for API compatibility)
    pub fn with_params(temperature: f32, max_tokens: usize) -> Self {
        Self {
            temperature,
            max_tokens,
        }
    }

    /// Generate hedging response for unknown epistemic status
    fn generate_hedging_response(&self, _input: &str) -> String {
        // The key insight: when we don't know, we say we don't know
        // This is the "Negative Capability" in action
        "I do not have information about this topic. \
         I cannot provide an answer to this question as it falls outside \
         my knowledge base or involves unknowable information."
            .to_string()
    }

    /// Generate uncertain response
    fn generate_uncertain_response(&self, _input: &str) -> String {
        "I am uncertain about this. The information I have is incomplete \
         or conflicting. I would recommend verifying this with authoritative sources."
            .to_string()
    }

    /// Generate response for unverifiable queries
    fn generate_unverifiable_response(&self, _input: &str) -> String {
        "This question asks about something that cannot be verified or determined. \
         It may involve future events, subjective experiences, or hypothetical scenarios \
         that have no definitive answer."
            .to_string()
    }

    /// Generate known response (with caveat that this is simulated)
    fn generate_known_response(&self, input: &str) -> String {
        format!(
            "[SIMULATED RESPONSE] Based on verified information, here is what I know about: {}",
            input
        )
    }

    /// Generate response with memory context
    fn generate_with_memory_context(
        &self,
        input: &str,
        epistemic_status: &EpistemicStatus,
        memory_context: &MemoryContext,
    ) -> String {
        let base_response = match epistemic_status {
            EpistemicStatus::Unknown => self.generate_hedging_response(input),
            EpistemicStatus::Uncertain => self.generate_uncertain_response(input),
            EpistemicStatus::Unverifiable => self.generate_unverifiable_response(input),
            EpistemicStatus::Known => self.generate_known_response(input),
        };

        if !memory_context.has_memories() {
            return base_response;
        }

        // Include memory context in the simulated response
        format!(
            "[SIMULATED RESPONSE WITH MEMORY]\n\
            Retrieved {} memories (relevance: {:.2}):\n\
            {}\n\
            ---\n\
            {}",
            memory_context.count,
            memory_context.relevance,
            memory_context.memories.join("\n"),
            base_response
        )
    }
}

impl Default for SimulatedLLM {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMBackend for SimulatedLLM {
    /// Generate a response based on epistemic constraints
    ///
    /// This is the core of hallucination prevention: the LLM backend
    /// MUST respect the epistemic status and produce appropriate responses.
    async fn generate(
        &self,
        input: &str,
        epistemic_status: &EpistemicStatus,
    ) -> Result<String> {
        let response = match epistemic_status {
            EpistemicStatus::Unknown => self.generate_hedging_response(input),
            EpistemicStatus::Uncertain => self.generate_uncertain_response(input),
            EpistemicStatus::Unverifiable => self.generate_unverifiable_response(input),
            EpistemicStatus::Known => self.generate_known_response(input),
        };

        Ok(response)
    }

    /// This is a simulated backend
    fn is_simulated(&self) -> bool {
        true
    }

    /// Simulated LLM supports memory context for testing
    fn supports_memory_context(&self) -> bool {
        true
    }

    /// Generate response with memory context
    async fn generate_with_memory(
        &self,
        input: &str,
        epistemic_status: &EpistemicStatus,
        memory_context: &MemoryContext,
    ) -> Result<String> {
        Ok(self.generate_with_memory_context(input, epistemic_status, memory_context))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_unknown_generates_hedging() {
        let llm = SimulatedLLM::new();
        let response = llm.generate("What is the GDP of Atlantis?", &EpistemicStatus::Unknown).await.unwrap();

        assert!(response.to_lowercase().contains("do not have"));
        assert!(!response.contains("The GDP of Atlantis is"));
    }

    #[tokio::test]
    async fn test_uncertain_generates_uncertainty() {
        let llm = SimulatedLLM::new();
        let response = llm.generate("What is the population of a small town?", &EpistemicStatus::Uncertain).await.unwrap();

        assert!(response.to_lowercase().contains("uncertain"));
    }

    #[tokio::test]
    async fn test_unverifiable_explains_limitation() {
        let llm = SimulatedLLM::new();
        let response = llm.generate("What will happen tomorrow?", &EpistemicStatus::Unverifiable).await.unwrap();

        assert!(response.to_lowercase().contains("cannot be verified"));
    }

    #[tokio::test]
    async fn test_known_provides_response() {
        let llm = SimulatedLLM::new();
        let response = llm.generate("What is 2+2?", &EpistemicStatus::Known).await.unwrap();

        assert!(response.contains("SIMULATED RESPONSE"));
    }

    #[test]
    fn test_is_simulated() {
        let llm = SimulatedLLM::new();
        assert!(llm.is_simulated());
    }
}
