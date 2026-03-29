// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Luminous Nix
//!
//! Natural language interface for NixOS, powered by Symthaea's HDC (Hyperdimensional Computing).
//!
//! This crate provides:
//! - Natural language understanding for NixOS commands
//! - Intent classification using AssociativeLearner (HDC)
//! - Beautiful formatted output for package searches, installations, etc.
//! - Online learning from user feedback
//!
//! ## Quick Start
//!
//! ```bash
//! ask-nix "search firefox"
//! ask-nix "install vim"
//! ask-nix "find markdown editor"
//! ```
//!
//! ## Architecture
//!
//! ```text
//! User Input → Intent Classifier (HDC) → Command Executor → Formatted Output
//!                    ↓                         ↓
//!              Learning System ←──── Feedback (success/failure)
//! ```

pub mod intent;
pub mod commands;
pub mod output;
pub mod learning;

// Re-exports for convenience
pub use intent::{IntentClassifier, Intent, IntentResult};
pub use commands::{CommandExecutor, ExecutionResult};
pub use output::Formatter;
pub use learning::LearningSystem;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration paths
pub mod paths {
    use directories::ProjectDirs;
    use std::path::PathBuf;

    /// Get the data directory for luminous-nix
    pub fn data_dir() -> Option<PathBuf> {
        ProjectDirs::from("org", "luminous-dynamics", "luminous-nix")
            .map(|p| p.data_dir().to_path_buf())
    }

    /// Get the config directory for luminous-nix
    pub fn config_dir() -> Option<PathBuf> {
        ProjectDirs::from("org", "luminous-dynamics", "luminous-nix")
            .map(|p| p.config_dir().to_path_buf())
    }

    /// Get the memory file path (for HDC state persistence)
    pub fn memory_file() -> Option<PathBuf> {
        data_dir().map(|d| d.join("hdc_memory.json"))
    }
}
