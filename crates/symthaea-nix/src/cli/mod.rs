// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Command-Line Interface
//!
//! The `nix-mind` CLI provides both natural language and subcommand
//! interfaces for NixOS management, backed by the cognitive core.

pub mod commands;
pub mod completions;
pub mod interactive;

pub use commands::{Cli, Command, OutputFormat};
pub use completions::generate_completions;
pub use interactive::{ConsciousRepl, ConsciousnessQuadrant, process_oneshot};
