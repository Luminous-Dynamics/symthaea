//! Isolated test harness for the active-sensing VoI planner.
//!
//! The implementation and canonical digest module are included directly from
//! the temporal-bench source tree so this crate tests the real code without
//! compiling unrelated, currently unqualified research modules.

#[path = "../../src/digest.rs"]
pub mod digest;

#[path = "../../src/active_sensing.rs"]
pub mod active_sensing;
