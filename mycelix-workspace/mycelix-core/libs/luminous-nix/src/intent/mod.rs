// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Intent classification using Symthaea's AssociativeLearner (HDC)
//!
//! This module provides natural language understanding for NixOS commands
//! using Hyperdimensional Computing for zero-shot generalization.

mod classifier;
mod patterns;

pub use classifier::{IntentClassifier, IntentResult};
pub use patterns::PatternMatcher;

use serde::{Deserialize, Serialize};

/// Recognized intents for NixOS operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Intent {
    /// Search for packages
    Search { query: String },

    /// Install a package
    Install { package: String },

    /// Remove a package
    Remove { package: String },

    /// List system generations
    ListGenerations,

    /// Rebuild the system
    Rebuild,

    /// Garbage collect
    GarbageCollect,

    /// Show system status
    Status,

    /// Show help
    Help,

    /// Unknown intent
    Unknown,
}

impl Intent {
    /// Get the action name for this intent (used in HDC learning)
    pub fn action_name(&self) -> &'static str {
        match self {
            Intent::Search { .. } => "search",
            Intent::Install { .. } => "install",
            Intent::Remove { .. } => "remove",
            Intent::ListGenerations => "list_generations",
            Intent::Rebuild => "rebuild",
            Intent::GarbageCollect => "garbage_collect",
            Intent::Status => "status",
            Intent::Help => "help",
            Intent::Unknown => "unknown",
        }
    }

    /// Create an intent from an action name and optional argument
    pub fn from_action(action: &str, arg: Option<String>) -> Self {
        match action {
            "search" => Intent::Search {
                query: arg.unwrap_or_default(),
            },
            "install" => Intent::Install {
                package: arg.unwrap_or_default(),
            },
            "remove" => Intent::Remove {
                package: arg.unwrap_or_default(),
            },
            "list_generations" => Intent::ListGenerations,
            "rebuild" => Intent::Rebuild,
            "garbage_collect" => Intent::GarbageCollect,
            "status" => Intent::Status,
            "help" => Intent::Help,
            _ => Intent::Unknown,
        }
    }
}

impl std::fmt::Display for Intent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Intent::Search { query } => write!(f, "search '{}'", query),
            Intent::Install { package } => write!(f, "install '{}'", package),
            Intent::Remove { package } => write!(f, "remove '{}'", package),
            Intent::ListGenerations => write!(f, "list generations"),
            Intent::Rebuild => write!(f, "rebuild system"),
            Intent::GarbageCollect => write!(f, "garbage collect"),
            Intent::Status => write!(f, "show status"),
            Intent::Help => write!(f, "show help"),
            Intent::Unknown => write!(f, "unknown"),
        }
    }
}
