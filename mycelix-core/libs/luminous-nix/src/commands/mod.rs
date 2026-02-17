//! Command execution for NixOS operations
//!
//! This module provides async executors for all NixOS commands,
//! optimized with JSON output parsing where available.

mod executor;
mod types;

pub use executor::CommandExecutor;
pub use types::*;
