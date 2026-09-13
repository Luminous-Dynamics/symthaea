//! Frozen external-analysis invocation and execution-receipt contracts.
//!
//! This crate closes one narrow research-integrity gap:
//!
//! ```text
//! frozen analysis-rule identity + result Analysis artifact
//!     != proof that the exact frozen invocation executed over the exact intended inputs
//! ```
//!
//! The protocol/result crates own preregistration and terminal result semantics. This crate adds:
//!
//! - a frozen invocation identity (executable + invocation/config + environment + ordered inputs);
//! - exact concrete input/output/run bindings;
//! - explicit execution and verifier terminal states;
//! - exact re-execution output matching;
//! - exact plan/receipt census for every external-rule endpoint;
//! - optional identity hooks to external custody receipts without duplicating custody policy.
//!
//! Identity is deliberately separate from chronology: #1946 must establish that the plan/source
//! plan digests entered the append-only research lineage before the relevant unblinding boundary.

mod error;
mod plan;
mod qualified;
mod receipt;
mod types;

pub use error::{ResearchAnalysisError, Result};
pub use plan::FrozenExternalAnalysisPlanV1;
pub use qualified::ExternalAnalysisQualifiedResultV1;
pub use receipt::{
    AnalysisInputBindingV1, AnalysisOutputBindingV1, FrozenAnalysisExecutionReceiptV1,
};
pub use types::{
    AnalysisExecutionStatus, AnalysisInputRole, AnalysisVerificationClass, AnalysisVerifierResult,
    AnalysisVerifierV1, FrozenAnalysisInputSpecV1,
};

#[cfg(test)]
mod tests;
