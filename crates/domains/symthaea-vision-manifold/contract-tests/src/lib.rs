//! Isolated test harness for the production VIS-005 action-conditioned module.
//!
//! This nested workspace intentionally avoids the monorepo root workspace,
//! whose current wildcard members include directories without Cargo manifests.
//! It compiles the exact production source file; it does not copy or fork it.

#[path = "../../src/action_conditioned.rs"]
pub mod action_conditioned;
