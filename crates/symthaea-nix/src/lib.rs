//! # Symthaea NixOS: A Conscious Mind for NixOS
//!
//! This crate implements a genuine NixOS world model using Symthaea's cognitive
//! architecture: active inference (Free Energy Principle), hierarchical predictive
//! processing, causal reasoning with HDC role markers, episodic memory with
//! Φ-weighted consolidation, and a 2D consciousness space (Φ × Confidence).
//!
//! ## Architecture Layers
//!
//! 1. **Parser** — Perception: raw Nix source → structured AST
//! 2. **Encoding** — Perception: structured data → HDC hypervectors
//! 3. **Mind** — Cognition: world model, active inference, causal graph
//! 4. **Observe** — Sensory input: system state observation
//! 5. **Action** — Motor output: Φ-gated command execution
//! 6. **Plugin** — Integration with full Symthaea brain
//! 7. **CLI** — Command-line interface
//! 8. **TUI** — Terminal UI with consciousness visualization

#![allow(deprecated)]
#![allow(rustdoc::broken_intra_doc_links)]

/// Local trait and type definitions for symthaea-nix.
///
/// These mirror traits from the main symthaea crate that the migrated
/// NixOS modules depend on, allowing symthaea-nix to compile standalone
/// while maintaining API compatibility.
pub mod traits;

/// Layer 1: Perception — Nix source parsing via tree-sitter
pub mod parser;

/// Layer 2: Perception — Structured data → HDC hypervectors
pub mod encoding;

/// Layer 3: Cognition — World model, active inference, causal graph
pub mod mind;

/// Layer 4: Sensory input — System state observation
pub mod observe;

/// Layer 5: Motor output — Φ-gated command execution
pub mod action;

/// Layer 6: Integration with full Symthaea brain
pub mod plugin;

/// Daemon ↔ TUI inter-process communication
pub mod ipc;

/// Layer 7: Command-line interface
#[cfg(feature = "cli")]
pub mod cli;

/// Layer 8: Terminal UI
#[cfg(feature = "tui")]
pub mod tui;

// Re-export key types at crate root
pub use action::executor::{ExecutionResult, NixOSCommand, NixOSExecutor, SafetyLevel};
pub use parser::nix_code_parser::NixCodeParser;
pub use parser::nix_parser::{NixConfig, NixOption, NixParser, NixValue};
pub use plugin::domain_plugin::NixOsPlugin;
