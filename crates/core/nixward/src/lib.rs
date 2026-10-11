// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
//! 5. **Action** — Explicit authority + cognitive telemetry + rollback-capable execution
//! 6. **Plugin** — Integration with full Symthaea brain
//! 7. **CLI** — Command-line interface
//! 8. **TUI** — Terminal UI with consciousness visualization

#![deny(unsafe_code)]
#![allow(deprecated)]

/// Local trait and type definitions for nixward.
///
/// These mirror traits from the main symthaea crate that the migrated
/// NixOS modules depend on, allowing nixward to compile standalone
/// while maintaining API compatibility.
pub mod traits;

// ── WASM-safe modules (compile on all targets) ──

/// Layer 2: Perception — Structured data → HDC hypervectors
pub mod encoding;

/// Layer 3: Cognition — World model, active inference, causal graph
pub mod mind;

/// App intelligence database — package search, migration analysis
pub mod app_database;

/// Sovereign NixOS configuration generator — hardware-aware, consciousness-coupled
pub mod sovereign_config;

/// Conversational NixOS installer — dialogue-driven config generation
pub mod sovereign_conversation;

// ── Native-only modules (require filesystem, process, async) ──

/// Layer 1: Perception — Nix source parsing via tree-sitter
#[cfg(feature = "native")]
pub mod parser;

/// Layer 4: Sensory input — System state observation
#[cfg(feature = "native")]
pub mod observe;

/// Layer 5: Motor output — explicit authority separated from cognition
#[cfg(feature = "native")]
pub mod action;

/// Layer 6: Integration with full Symthaea brain
#[cfg(feature = "native")]
pub mod plugin;

/// Proactive NixOS support: health checks, watchdog, predictions, knowledge base
#[cfg(feature = "native")]
pub mod support;

/// Daemon ↔ TUI inter-process communication
pub mod ipc;

/// Layer 7: Command-line interface
#[cfg(feature = "cli")]
pub mod cli;

/// Layer 8: Terminal UI
#[cfg(feature = "tui")]
pub mod tui;

/// Production observability
#[cfg(feature = "observability")]
pub mod observability;

// Re-export key types (native only — these depend on parser/action/plugin)
#[cfg(feature = "native")]
pub use action::executor::{ExecutionResult, NixOSCommand, NixOSExecutor, SafetyLevel};
#[cfg(feature = "native")]
pub use action::execution_context::{
    AuthorityContext, AuthoritySource, CognitiveContext, ExecutionContext, PhiMeasurement,
    EXECUTION_CONTEXT_SCHEMA_VERSION,
};
#[cfg(feature = "native")]
pub use parser::nix_code_parser::NixCodeParser;
#[cfg(feature = "native")]
pub use parser::nix_parser::{NixConfig, NixOption, NixParser, NixValue};
#[cfg(feature = "native")]
pub use plugin::domain_plugin::NixOsPlugin;
