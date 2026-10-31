//! Fail-closed validation for the frozen Symthaea architecture-engineering v0.1 matrix.
//!
//! This crate validates the *design fixture*, not experiment results. Measured results
//! belong in separate, provenance-bearing run receipts; this frozen matrix intentionally
//! requires every measurement to remain null and every record to remain NOT_RUN.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use serde_json::Value;

pub mod paired_evaluation;
pub mod synthetic_search;
pub mod multi_fidelity;
pub mod multi_fidelity_benchmark;
pub mod analytical_benchmark;
pub mod direction_board;
pub mod direction_board_benchmark;
pub mod event_tape;
pub mod coral_baseline;
pub mod resource_pareto;
