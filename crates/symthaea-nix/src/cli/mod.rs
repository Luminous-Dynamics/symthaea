//! Command-Line Interface
//!
//! The `nix-mind` CLI provides both natural language and subcommand
//! interfaces for NixOS management, backed by the cognitive core.

pub mod commands;
pub mod interactive;
pub mod completions;

pub use commands::{Cli, Command, OutputFormat};
pub use interactive::{ConsciousRepl, ConsciousnessQuadrant, process_oneshot};
pub use completions::generate_completions;
