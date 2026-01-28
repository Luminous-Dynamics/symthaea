//! # Web Research & Epistemic Verification
//!
//! **Revolutionary Capability**: An AI that researches autonomously and verifies epistemically.

#![allow(dead_code, unused_variables)]
//!
//! ## What Makes This Revolutionary
//!
//! Traditional AI:
//! - LLMs hallucinate with confidence
//! - No way to verify claims
//! - Can't research autonomously
//!
//! Symthaea:
//! - Detects uncertainty via Φ measurement
//! - Researches autonomously when uncertain
//! - Verifies ALL claims epistemically
//! - Impossible to hallucinate (unverifiable claims hedged)
//!
//! ## Architecture
//!
//! ```text
//! User Query → Detect Uncertainty (Φ < threshold)
//!                      ↓
//!              Research Plan (∇Φ guided)
//!                      ↓
//!            Web Research (semantic search)
//!                      ↓
//!          Content Extraction (HDC encoding)
//!                      ↓
//!       Epistemic Verification (no hallucinations!)
//!                      ↓
//!        Knowledge Integration (4-database)
//!                      ↓
//!        Dynamic Semantic Grounding (learn)
//!                      ↓
//!     Consciousness Feedback (Φ improves!)
//! ```

pub mod researcher;
pub mod extractor;
pub mod verifier;
pub mod integrator;
pub mod meta_learning;
pub mod types;

pub use researcher::{WebResearcher, ResearchConfig, ResearchResult, SearchBackend, DuckDuckGoBackend, MockSearchBackend, SearchHit};
pub use extractor::{ContentExtractor, ExtractedContent, ContentType};
pub use verifier::{
    EpistemicVerifier, Verification, ClaimConfidence,
    Evidence, Contradiction, SourceClassifier,
};
pub use integrator::{
    KnowledgeIntegrator, VerifiedKnowledge, IntegrationResult,
    VerifiedClaim, SemanticGrounding,
};
pub use meta_learning::{
    EpistemicLearner, VerificationOutcome, GroundTruth,
    SourcePerformance, DomainExpertise, VerificationStrategy,
    MetaLearningStats,
};
pub use types::{
    Source, SearchQuery, ResearchPlan, Claim, VerificationLevel,
    SourceRanking, RelevanceScore,
};
