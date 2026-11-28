//! Compile the production ResearchSource, WebResearchResult and EpistemicVerifier
//! modules by path so regression tests execute the repository implementation.

#[path = "../../../src/web_research/types.rs"]
pub mod types;

#[path = "../../../src/web_research/verifier.rs"]
pub mod verifier;
