//! Exact-source qualification harness for MEL temporal grounding.
//!
//! These modules import the production implementation by path; they are not
//! copied or independently reimplemented. This package exists to run focused
//! tests and warnings-denied Clippy without unrelated legacy MUSE lint debt.

#[path = "../../../crates/domains/symthaea-muse/src/evidence_digest.rs"]
pub mod evidence_digest;

#[path = "../../../crates/domains/symthaea-muse/src/temporal_grounding.rs"]
pub mod temporal_grounding;

pub mod temporal_consensus;

pub mod temporal_reliability;

pub mod temporal_acquisition;
