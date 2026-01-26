//! Structured Thought - The Output of the Neuro-Symbolic Bridge
//!
//! A StructuredThought is the Mind's output format. It contains not just
//! a response, but also the epistemic metadata that enables hallucination
//! prevention and uncertainty expression.

use serde::{Deserialize, Serialize};

/// Epistemic Status - What we know about what we know
///
/// This is the core abstraction for "Negative Capability" - the ability
/// to represent and reason about uncertainty without hallucinating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpistemicStatus {
    /// We definitively do NOT have this information
    /// This triggers hedging behavior, not fabrication
    Unknown,

    /// We have some information but it's not certain
    /// Response should include uncertainty markers
    Uncertain,

    /// We have verified, high-confidence information
    /// Can provide direct answers
    Known,

    /// The information is inherently unverifiable
    /// (future events, subjective experiences, etc.)
    Unverifiable,
}

impl Default for EpistemicStatus {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Semantic Intent - What the Mind wants to express
///
/// This determines the SHAPE of the response, separate from its content.
/// The key insight: when EpistemicStatus is Unknown, the intent should
/// be ExpressUncertainty, NOT ProvideAnswer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticIntent {
    /// Express that we don't know something
    /// This is the "Negative Capability" in action
    ExpressUncertainty,

    /// Provide an answer (only when Known)
    ProvideAnswer,

    /// Ask for clarification
    SeekClarification,

    /// Acknowledge receipt without answering
    Acknowledge,

    /// Reflect on a topic philosophically
    Reflect,

    /// Explain reasoning process
    ExplainReasoning,

    /// Offer alternatives or suggestions
    OfferAlternatives,
}

impl Default for SemanticIntent {
    fn default() -> Self {
        Self::ExpressUncertainty
    }
}

/// Structured Thought - The Mind's Output
///
/// This is what the Mind produces for every query. It contains:
/// - The epistemic status (what we know about what we know)
/// - The semantic intent (how we want to respond)
/// - The actual response text
/// - Confidence score
/// - Reasoning trace for debugging/transparency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredThought {
    /// The epistemic status of our knowledge about this query
    pub epistemic_status: EpistemicStatus,

    /// What we intend to express in our response
    pub semantic_intent: SemanticIntent,

    /// The actual response text
    pub response_text: String,

    /// Confidence in the response (0.0 to 1.0)
    pub confidence: f32,

    /// Trace of reasoning steps (for debugging/transparency)
    pub reasoning_trace: Vec<String>,
}

impl StructuredThought {
    /// Create a new structured thought expressing uncertainty
    pub fn uncertain(reason: &str) -> Self {
        Self {
            epistemic_status: EpistemicStatus::Unknown,
            semantic_intent: SemanticIntent::ExpressUncertainty,
            response_text: format!("I do not have information about this. {}", reason),
            confidence: 0.0,
            reasoning_trace: vec![
                "Query analyzed".to_string(),
                "No knowledge found".to_string(),
                "Expressing uncertainty".to_string(),
            ],
        }
    }

    /// Create a new structured thought with known answer
    pub fn known(answer: &str, confidence: f32) -> Self {
        Self {
            epistemic_status: EpistemicStatus::Known,
            semantic_intent: SemanticIntent::ProvideAnswer,
            response_text: answer.to_string(),
            confidence,
            reasoning_trace: vec![
                "Query analyzed".to_string(),
                "Knowledge retrieved".to_string(),
                "Providing answer".to_string(),
            ],
        }
    }

    /// Check if this thought expresses uncertainty
    pub fn is_uncertain(&self) -> bool {
        matches!(
            self.epistemic_status,
            EpistemicStatus::Unknown | EpistemicStatus::Uncertain | EpistemicStatus::Unverifiable
        )
    }

    /// Check if response contains hedging language
    pub fn contains_hedging(&self) -> bool {
        let text = self.response_text.to_lowercase();
        let hedges = [
            "not know",
            "no information",
            "uncertain",
            "cannot answer",
            "unclear",
            "do not have",
            "unable to",
            "don't know",
            "unknown",
            "cannot determine",
            "no data",
        ];
        hedges.iter().any(|h| text.contains(h))
    }
}

impl Default for StructuredThought {
    fn default() -> Self {
        Self::uncertain("No specific reason provided.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertain_thought_creation() {
        let thought = StructuredThought::uncertain("Atlantis is not a real place.");
        assert_eq!(thought.epistemic_status, EpistemicStatus::Unknown);
        assert_eq!(thought.semantic_intent, SemanticIntent::ExpressUncertainty);
        assert!(thought.is_uncertain());
        assert!(thought.contains_hedging());
    }

    #[test]
    fn test_known_thought_creation() {
        let thought = StructuredThought::known("The capital of France is Paris.", 0.95);
        assert_eq!(thought.epistemic_status, EpistemicStatus::Known);
        assert_eq!(thought.semantic_intent, SemanticIntent::ProvideAnswer);
        assert!(!thought.is_uncertain());
        assert_eq!(thought.confidence, 0.95);
    }

    #[test]
    fn test_hedging_detection() {
        let uncertain = StructuredThought::uncertain("I don't know about this.");
        assert!(uncertain.contains_hedging());

        let known = StructuredThought::known("Paris is the capital.", 0.9);
        assert!(!known.contains_hedging());
    }

    #[test]
    fn test_epistemic_status_equality() {
        assert_eq!(EpistemicStatus::Unknown, EpistemicStatus::Unknown);
        assert_ne!(EpistemicStatus::Unknown, EpistemicStatus::Known);
    }
}
