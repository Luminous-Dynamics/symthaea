//! Causal Reasoning for Symthaea
//!
//! Pearl's do-calculus, counterfactual reasoning, and causal emergence analysis.
//!
//! - [`causal_calculus`]: Structural Causal Models, do-calculus rules, interventional queries
//! - [`causal_emergence`]: Hoel's Effective Information and causal emergence measurement
//! - [`counterfactual`]: Backdoor/frontdoor identification, HDC graph surgery, semantic roles

#![allow(clippy::needless_range_loop)]

pub mod causal_calculus;
pub mod causal_emergence;

#[cfg(feature = "counterfactual")]
pub mod counterfactual;
