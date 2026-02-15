//! Command-Line Interface
//!
//! The `nix-mind` CLI provides both natural language and subcommand
//! interfaces for NixOS management, backed by the cognitive core.

pub mod commands;
pub mod completions;
pub mod interactive;

pub use commands::{Cli, Command, OutputFormat};
pub use completions::generate_completions;
pub use interactive::{process_oneshot, ConsciousRepl, ConsciousnessQuadrant};
