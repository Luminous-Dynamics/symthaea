//! Standalone compile/test harness for the exact architecture-engineering source files.
//!
//! This isolates the SciLaws audit from unrelated legacy modules in the
//! stacked architecture-engineering package. It imports the shared Pareto
//! engine and the actual SciLaws audit source by path; neither implementation
//! is copied or forked here.

#[path = "../../../crates/domains/symthaea-architecture-engineering/src/pareto_objectives.rs"]
pub mod pareto_objectives;

#[path = "../../../crates/domains/symthaea-architecture-engineering/src/scilaws_candidate_selection.rs"]
pub mod scilaws_candidate_selection;
