//! SympoeticBench's deterministic validation and measurement core.
//!
//! This crate deliberately does not contain a model, an LLM judge, or a
//! human-subject scoring shortcut. It validates benchmark artifacts and
//! computes only metrics whose inputs and assumptions are explicit.

pub mod metrics;
pub mod model;
pub mod validate;

pub use model::{Scenario, Scorecard, Trace};
pub use validate::{
    CorpusValidation, TraceValidation, validate_corpus, validate_scenario,
    validate_scorecard, validate_trace,
};
