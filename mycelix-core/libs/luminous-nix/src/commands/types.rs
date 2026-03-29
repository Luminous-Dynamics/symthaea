// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Types for command execution results

use serde::{Deserialize, Serialize};

/// Result of executing a command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Whether the command succeeded
    pub success: bool,

    /// The command that was executed
    pub command: String,

    /// Standard output
    pub stdout: String,

    /// Standard error
    pub stderr: String,

    /// Exit code
    pub exit_code: Option<i32>,

    /// Execution time in milliseconds
    pub duration_ms: u64,

    /// Whether this was a dry run
    pub dry_run: bool,
}

impl ExecutionResult {
    /// Create a success result
    pub fn success(command: String, stdout: String, duration_ms: u64) -> Self {
        Self {
            success: true,
            command,
            stdout,
            stderr: String::new(),
            exit_code: Some(0),
            duration_ms,
            dry_run: false,
        }
    }

    /// Create a failure result
    pub fn failure(command: String, stderr: String, exit_code: i32, duration_ms: u64) -> Self {
        Self {
            success: false,
            command,
            stdout: String::new(),
            stderr,
            exit_code: Some(exit_code),
            duration_ms,
            dry_run: false,
        }
    }

    /// Create a dry run result
    pub fn dry_run(command: String) -> Self {
        Self {
            success: true,
            command,
            stdout: String::new(),
            stderr: String::new(),
            exit_code: None,
            duration_ms: 0,
            dry_run: true,
        }
    }
}

/// A package search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Packages found
    pub packages: Vec<PackageInfo>,

    /// Total count (may be more than returned)
    pub total_count: usize,

    /// Query that was searched
    pub query: String,

    /// Whether this was a dry run
    pub dry_run: bool,
}

/// Information about a Nix package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageInfo {
    /// Package attribute path (e.g., "nixpkgs.firefox")
    pub attr_path: String,

    /// Package name
    pub name: String,

    /// Package version
    pub version: String,

    /// Package description
    pub description: Option<String>,

    /// Homepage URL
    pub homepage: Option<String>,

    /// License
    pub license: Option<String>,
}

impl PackageInfo {
    /// Create from JSON output of `nix search --json`
    pub fn from_nix_json(attr_path: &str, json: &serde_json::Value) -> Option<Self> {
        let name = json.get("pname")?.as_str()?.to_string();
        let version = json.get("version")?.as_str()?.to_string();
        let description = json
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Some(Self {
            attr_path: attr_path.to_string(),
            name,
            version,
            description,
            homepage: None,
            license: None,
        })
    }
}

/// A system generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generation {
    /// Generation number
    pub number: u32,

    /// Date/time string
    pub date: String,

    /// NixOS version
    pub nixos_version: Option<String>,

    /// Kernel version
    pub kernel_version: Option<String>,

    /// Configuration revision (if available)
    pub config_revision: Option<String>,

    /// Whether this is the current generation
    pub current: bool,
}

/// List of generations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationList {
    /// All generations
    pub generations: Vec<Generation>,

    /// Current generation number
    pub current: u32,

    /// Whether this was a dry run
    pub dry_run: bool,
}

/// System status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    /// NixOS version
    pub nixos_version: String,

    /// Kernel version
    pub kernel_version: String,

    /// Current generation number
    pub current_generation: u32,

    /// Number of generations
    pub total_generations: u32,

    /// Nix store size (human readable)
    pub store_size: Option<String>,

    /// Last rebuild time
    pub last_rebuild: Option<String>,
}
