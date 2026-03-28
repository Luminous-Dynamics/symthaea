//! Causal Identification
//!
//! Implements Pearl's do-calculus rules for causal identification:
//! - Rule 1: Insertion/deletion of observations (backdoor/frontdoor criteria)
//! - Rule 2: Action/observation exchange
//! - Rule 3: Insertion/deletion of actions
//!
//! Returns `Identified`, `Unidentified`, or `AssumptionRequired` — never overclaims.

pub mod dag;
pub mod discovery;
pub mod estimation;
pub mod id_algorithm;
pub mod reasoner;

#[cfg(test)]
mod tests;

// Re-export everything for backward compatibility
pub use dag::*;
pub use discovery::*;
pub use estimation::*;
pub use id_algorithm::*;
pub use reasoner::*;
