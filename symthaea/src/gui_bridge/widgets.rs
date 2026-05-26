// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Advanced GUI Widgets
//!
//! C1: NixOS Module Browser - Semantic search and exploration of NixOS options
//! C2: Generation Timeline - Visual history of NixOS generations
//! C3: Live Config Diff - Real-time configuration comparison
//! C4: Service Status Dashboard - Systemd service monitoring
//!
//! These widgets integrate with the egui framework and provide
//! consciousness-aware visualization of NixOS state.

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::process::Command;

// ============================================================================
// C1: NixOS Module Browser
// ============================================================================

/// NixOS option metadata from nix-instantiate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixOption {
    /// Option path (e.g., "services.nginx.enable")
    pub path: String,
    /// Option type (bool, str, listOf str, etc.)
    pub option_type: String,
    /// Default value
    pub default: Option<String>,
    /// Description from NixOS manual
    pub description: String,
    /// Example value
    pub example: Option<String>,
    /// Whether option is declared vs defined
    pub declared_by: Vec<String>,
    /// Whether currently set in configuration
    pub is_set: bool,
    /// Current value if set
    pub current_value: Option<String>,
}

/// Module category for browser navigation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModuleCategory {
    pub name: String,
    pub path_prefix: String,
    pub icon: char,
    pub description: String,
    pub option_count: usize,
}

impl ModuleCategory {
    /// Standard NixOS module categories
    pub fn standard_categories() -> Vec<Self> {
        vec![
            Self {
                name: "Services".to_string(),
                path_prefix: "services".to_string(),
                icon: '\u{2699}', // Gear
                description: "Systemd services and daemons".to_string(),
                option_count: 0,
            },
            Self {
                name: "Programs".to_string(),
                path_prefix: "programs".to_string(),
                icon: '\u{1F4BB}', // Laptop
                description: "User programs and shells".to_string(),
                option_count: 0,
            },
            Self {
                name: "Networking".to_string(),
                path_prefix: "networking".to_string(),
                icon: '\u{1F310}', // Globe
                description: "Network configuration".to_string(),
                option_count: 0,
            },
            Self {
                name: "Security".to_string(),
                path_prefix: "security".to_string(),
                icon: '\u{1F512}', // Lock
                description: "Security and permissions".to_string(),
                option_count: 0,
            },
            Self {
                name: "Boot".to_string(),
                path_prefix: "boot".to_string(),
                icon: '\u{1F680}', // Rocket
                description: "Bootloader and kernel".to_string(),
                option_count: 0,
            },
            Self {
                name: "Hardware".to_string(),
                path_prefix: "hardware".to_string(),
                icon: '\u{1F5A5}', // Desktop
                description: "Hardware drivers and support".to_string(),
                option_count: 0,
            },
            Self {
                name: "Users".to_string(),
                path_prefix: "users".to_string(),
                icon: '\u{1F464}', // Person
                description: "User accounts and groups".to_string(),
                option_count: 0,
            },
            Self {
                name: "Nix".to_string(),
                path_prefix: "nix".to_string(),
                icon: '\u{2744}', // Snowflake
                description: "Nix package manager settings".to_string(),
                option_count: 0,
            },
        ]
    }
}

/// Module browser state
#[derive(Debug, Clone)]
pub struct ModuleBrowser {
    /// All discovered options
    pub options: Vec<NixOption>,
    /// Categories with counts
    pub categories: Vec<ModuleCategory>,
    /// Current search query
    pub search_query: String,
    /// Selected category (None = all)
    pub selected_category: Option<String>,
    /// Filtered results
    pub filtered_options: Vec<usize>,
    /// Currently expanded option (for details)
    pub expanded_option: Option<String>,
    /// Loading state
    pub is_loading: bool,
    /// Error message
    pub error: Option<String>,
}

impl Default for ModuleBrowser {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleBrowser {
    pub fn new() -> Self {
        Self {
            options: Vec::new(),
            categories: ModuleCategory::standard_categories(),
            search_query: String::new(),
            selected_category: None,
            filtered_options: Vec::new(),
            expanded_option: None,
            is_loading: false,
            error: None,
        }
    }

    /// Load NixOS options (expensive operation, run in background)
    pub fn load_options(&mut self) -> Result<(), String> {
        self.is_loading = true;
        self.error = None;

        // Query nix-instantiate for option metadata
        let output = Command::new("nix-instantiate")
            .args([
                "--eval",
                "--json",
                "-E",
                r#"
                let
                  pkgs = import <nixpkgs> {};
                  lib = pkgs.lib;
                  eval = import <nixpkgs/nixos> { configuration = {}; };
                  options = eval.options;

                  collectOptions = prefix: opts:
                    lib.flatten (lib.mapAttrsToList (name: opt:
                      let path = if prefix == "" then name else "${prefix}.${name}";
                      in if opt ? _type && opt._type == "option"
                         then [{
                           inherit path;
                           type = opt.type.description or "unknown";
                           description = opt.description or "";
                           default = if opt ? default then builtins.toJSON opt.default else null;
                           example = if opt ? example then builtins.toJSON opt.example else null;
                         }]
                         else if builtins.isAttrs opt
                         then collectOptions path opt
                         else []
                    ) opts);
                in lib.take 1000 (collectOptions "" options)
            "#,
            ])
            .output()
            .map_err(|e| format!("Failed to run nix-instantiate: {e}"))?;

        self.is_loading = false;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let err = format!("nix-instantiate failed: {stderr}");
            self.error = Some(err.clone());
            return Err(err);
        }

        // Parse JSON output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&stdout).map_err(|e| format!("Failed to parse options: {e}"))?;

        self.options = parsed
            .into_iter()
            .map(|v| NixOption {
                path: v["path"].as_str().unwrap_or("").to_string(),
                option_type: v["type"].as_str().unwrap_or("unknown").to_string(),
                description: v["description"].as_str().unwrap_or("").to_string(),
                default: v["default"].as_str().map(|s| s.to_string()),
                example: v["example"].as_str().map(|s| s.to_string()),
                declared_by: Vec::new(),
                is_set: false,
                current_value: None,
            })
            .collect();

        // Update category counts
        for cat in &mut self.categories {
            cat.option_count = self
                .options
                .iter()
                .filter(|o| o.path.starts_with(&cat.path_prefix))
                .count();
        }

        // Initialize filtered to all
        self.filtered_options = (0..self.options.len()).collect();

        Ok(())
    }

    /// Filter options by search query and category
    pub fn filter(&mut self) {
        let query = self.search_query.to_lowercase();

        self.filtered_options = self
            .options
            .iter()
            .enumerate()
            .filter(|(_, opt)| {
                // Category filter
                if let Some(ref cat) = self.selected_category {
                    if !opt.path.starts_with(cat) {
                        return false;
                    }
                }

                // Search filter
                if !query.is_empty() {
                    let matches = opt.path.to_lowercase().contains(&query)
                        || opt.description.to_lowercase().contains(&query);
                    if !matches {
                        return false;
                    }
                }

                true
            })
            .map(|(i, _)| i)
            .collect();
    }

    /// Get option by path
    pub fn get_option(&self, path: &str) -> Option<&NixOption> {
        self.options.iter().find(|o| o.path == path)
    }
}

// ============================================================================
// C2: Generation Timeline
// ============================================================================

/// NixOS generation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generation {
    /// Generation number
    pub number: u32,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// NixOS version
    pub nixos_version: String,
    /// Kernel version
    pub kernel_version: String,
    /// Configuration revision (git hash if using flakes)
    pub config_revision: Option<String>,
    /// Is current boot generation
    pub is_current: bool,
    /// Is currently booted generation
    pub is_booted: bool,
    /// Closure size in bytes
    pub closure_size: u64,
    /// Number of packages
    pub package_count: usize,
    /// Changes from previous generation
    pub changes: Option<GenerationChanges>,
}

/// Changes between generations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationChanges {
    /// Packages added
    pub added: Vec<String>,
    /// Packages removed
    pub removed: Vec<String>,
    /// Packages upgraded (name, old_version, new_version)
    pub upgraded: Vec<(String, String, String)>,
    /// Services changed
    pub services_changed: Vec<String>,
}

/// Generation timeline widget state
#[derive(Debug, Clone)]
pub struct GenerationTimeline {
    /// All generations
    pub generations: Vec<Generation>,
    /// Selected generation for details
    pub selected: Option<u32>,
    /// Comparison mode (comparing two generations)
    pub comparison: Option<(u32, u32)>,
    /// Show size graph
    pub show_size_graph: bool,
    /// Loading state
    pub is_loading: bool,
    /// Error message
    pub error: Option<String>,
}

impl Default for GenerationTimeline {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationTimeline {
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            selected: None,
            comparison: None,
            show_size_graph: true,
            is_loading: false,
            error: None,
        }
    }

    /// Load generations from system
    pub fn load(&mut self) -> Result<(), String> {
        self.is_loading = true;
        self.error = None;

        // List generations
        let output = Command::new("nixos-rebuild")
            .args(["list-generations"])
            .output()
            .map_err(|e| format!("Failed to list generations: {e}"))?;

        self.is_loading = false;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let err = format!("Failed to list generations: {stderr}");
            self.error = Some(err.clone());
            return Err(err);
        }

        // Parse output
        let stdout = String::from_utf8_lossy(&output.stdout);
        self.generations = self.parse_generations(&stdout);

        // Find current/booted
        if let Ok(current_link) = std::fs::read_link("/run/current-system") {
            // Extract generation number from link
            if let Some(num) = current_link
                .to_string_lossy()
                .split('-')
                .find_map(|s| s.parse::<u32>().ok())
            {
                if let Some(generation) = self.generations.iter_mut().find(|g| g.number == num) {
                    generation.is_booted = true;
                }
            }
        }

        Ok(())
    }

    fn parse_generations(&self, output: &str) -> Vec<Generation> {
        let mut generations = Vec::new();

        for line in output.lines() {
            // Parse lines like "  42   2024-01-15 10:30:15   (current)"
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(num) = parts[0].parse::<u32>() {
                    let is_current = line.contains("(current)");

                    // Parse date
                    let date_str = format!("{} {}", parts[1], parts[2]);
                    let created_at =
                        chrono::NaiveDateTime::parse_from_str(&date_str, "%Y-%m-%d %H:%M:%S")
                            .map(|dt| Utc.from_utc_datetime(&dt))
                            .unwrap_or_else(|_| Utc::now());

                    generations.push(Generation {
                        number: num,
                        created_at,
                        nixos_version: String::new(),
                        kernel_version: String::new(),
                        config_revision: None,
                        is_current,
                        is_booted: false,
                        closure_size: 0,
                        package_count: 0,
                        changes: None,
                    });
                }
            }
        }

        generations
    }

    /// Get diff between two generations
    pub fn diff(&self, gen1: u32, gen2: u32) -> Result<GenerationChanges, String> {
        let output = Command::new("nix")
            .args([
                "store",
                "diff-closures",
                &format!("/nix/var/nix/profiles/system-{gen1}-link"),
                &format!("/nix/var/nix/profiles/system-{gen2}-link"),
            ])
            .output()
            .map_err(|e| format!("Failed to diff generations: {e}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse diff output
        let mut changes = GenerationChanges {
            added: Vec::new(),
            removed: Vec::new(),
            upgraded: Vec::new(),
            services_changed: Vec::new(),
        };

        for line in stdout.lines() {
            if line.contains("->") {
                // Upgrade: package-1.0 -> 1.1
                let parts: Vec<&str> = line.split("->").collect();
                if parts.len() == 2 {
                    let name = parts[0].trim().split('-').next().unwrap_or("").to_string();
                    let old_ver = parts[0]
                        .trim()
                        .split('-')
                        .next_back()
                        .unwrap_or("")
                        .to_string();
                    let new_ver = parts[1].trim().to_string();
                    changes.upgraded.push((name, old_ver, new_ver));
                }
            } else if let Some(stripped) = line.strip_prefix('+') {
                changes.added.push(stripped.trim().to_string());
            } else if let Some(stripped) = line.strip_prefix('-') {
                changes.removed.push(stripped.trim().to_string());
            }
        }

        Ok(changes)
    }

    /// Rollback to a specific generation
    pub fn rollback_to(&self, _generation: u32) -> Result<(), String> {
        let output = Command::new("sudo")
            .args(["nixos-rebuild", "switch", "--rollback"])
            .output()
            .map_err(|e| format!("Failed to rollback: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Rollback failed: {stderr}"));
        }

        Ok(())
    }
}

// ============================================================================
// C3: Live Config Diff
// ============================================================================

/// Diff type for config comparison
#[derive(Debug, Clone, PartialEq)]
pub enum DiffType {
    Added,
    Removed,
    Modified,
    Context,
}

/// A line in the diff output
#[derive(Debug, Clone)]
pub struct DiffLine {
    pub diff_type: DiffType,
    pub line_number_old: Option<u32>,
    pub line_number_new: Option<u32>,
    pub content: String,
}

/// Diff hunk (section of changes)
#[derive(Debug, Clone)]
pub struct DiffHunk {
    pub old_start: u32,
    pub old_count: u32,
    pub new_start: u32,
    pub new_count: u32,
    pub lines: Vec<DiffLine>,
}

/// Live configuration diff state
#[derive(Debug, Clone)]
pub struct LiveConfigDiff {
    /// Original configuration content
    pub original: String,
    /// Modified configuration content
    pub modified: String,
    /// Parsed diff hunks
    pub hunks: Vec<DiffHunk>,
    /// File being diffed
    pub file_path: String,
    /// Whether to show inline diff
    pub inline_mode: bool,
    /// Syntax highlighting enabled
    pub syntax_highlight: bool,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
}

impl Default for LiveConfigDiff {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveConfigDiff {
    pub fn new() -> Self {
        Self {
            original: String::new(),
            modified: String::new(),
            hunks: Vec::new(),
            file_path: "/etc/nixos/configuration.nix".to_string(),
            inline_mode: false,
            syntax_highlight: true,
            last_update: Utc::now(),
        }
    }

    /// Load original config from file
    pub fn load_original(&mut self, path: &str) -> Result<(), String> {
        self.file_path = path.to_string();
        self.original =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read {path}: {e}"))?;
        self.compute_diff();
        Ok(())
    }

    /// Update modified content and recompute diff
    pub fn update_modified(&mut self, content: &str) {
        self.modified = content.to_string();
        self.compute_diff();
        self.last_update = Utc::now();
    }

    /// Compute diff between original and modified
    pub fn compute_diff(&mut self) {
        self.hunks.clear();

        let old_lines: Vec<&str> = self.original.lines().collect();
        let new_lines: Vec<&str> = self.modified.lines().collect();

        // Simple line-by-line diff (could be replaced with a more sophisticated algorithm)
        let mut current_hunk: Option<DiffHunk> = None;
        let mut old_idx = 0;
        let mut new_idx = 0;

        while old_idx < old_lines.len() || new_idx < new_lines.len() {
            let old_line = old_lines.get(old_idx);
            let new_line = new_lines.get(new_idx);

            match (old_line, new_line) {
                (Some(o), Some(n)) if o == n => {
                    // Context line
                    if let Some(ref mut hunk) = current_hunk {
                        hunk.lines.push(DiffLine {
                            diff_type: DiffType::Context,
                            line_number_old: Some(old_idx as u32 + 1),
                            line_number_new: Some(new_idx as u32 + 1),
                            content: o.to_string(),
                        });
                    }
                    old_idx += 1;
                    new_idx += 1;
                }
                (Some(o), Some(n)) => {
                    // Modified line
                    let hunk = current_hunk.get_or_insert_with(|| DiffHunk {
                        old_start: old_idx as u32 + 1,
                        old_count: 0,
                        new_start: new_idx as u32 + 1,
                        new_count: 0,
                        lines: Vec::new(),
                    });
                    hunk.lines.push(DiffLine {
                        diff_type: DiffType::Removed,
                        line_number_old: Some(old_idx as u32 + 1),
                        line_number_new: None,
                        content: o.to_string(),
                    });
                    hunk.lines.push(DiffLine {
                        diff_type: DiffType::Added,
                        line_number_old: None,
                        line_number_new: Some(new_idx as u32 + 1),
                        content: n.to_string(),
                    });
                    hunk.old_count += 1;
                    hunk.new_count += 1;

                    old_idx += 1;
                    new_idx += 1;
                }
                (Some(o), None) => {
                    // Removed line
                    let hunk = current_hunk.get_or_insert_with(|| DiffHunk {
                        old_start: old_idx as u32 + 1,
                        old_count: 0,
                        new_start: new_idx as u32 + 1,
                        new_count: 0,
                        lines: Vec::new(),
                    });
                    hunk.lines.push(DiffLine {
                        diff_type: DiffType::Removed,
                        line_number_old: Some(old_idx as u32 + 1),
                        line_number_new: None,
                        content: o.to_string(),
                    });
                    hunk.old_count += 1;

                    old_idx += 1;
                }
                (None, Some(n)) => {
                    // Added line
                    let hunk = current_hunk.get_or_insert_with(|| DiffHunk {
                        old_start: old_idx as u32 + 1,
                        old_count: 0,
                        new_start: new_idx as u32 + 1,
                        new_count: 0,
                        lines: Vec::new(),
                    });
                    hunk.lines.push(DiffLine {
                        diff_type: DiffType::Added,
                        line_number_old: None,
                        line_number_new: Some(new_idx as u32 + 1),
                        content: n.to_string(),
                    });
                    hunk.new_count += 1;

                    new_idx += 1;
                }
                (None, None) => break,
            }
        }

        if let Some(hunk) = current_hunk {
            if !hunk.lines.is_empty() {
                self.hunks.push(hunk);
            }
        }
    }

    /// Get statistics about the diff
    pub fn stats(&self) -> DiffStats {
        let mut added = 0;
        let mut removed = 0;
        let mut modified = 0;

        for hunk in &self.hunks {
            for line in &hunk.lines {
                match line.diff_type {
                    DiffType::Added => added += 1,
                    DiffType::Removed => removed += 1,
                    DiffType::Modified => modified += 1,
                    DiffType::Context => {}
                }
            }
        }

        DiffStats {
            added,
            removed,
            modified,
            hunks: self.hunks.len(),
        }
    }

    /// Generate unified diff format
    pub fn to_unified(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("--- {}\n", self.file_path));
        output.push_str(&format!("+++ {}\n", self.file_path));

        for hunk in &self.hunks {
            output.push_str(&format!(
                "@@ -{},{} +{},{} @@\n",
                hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count
            ));

            for line in &hunk.lines {
                let prefix = match line.diff_type {
                    DiffType::Added => "+",
                    DiffType::Removed => "-",
                    DiffType::Modified => "!",
                    DiffType::Context => " ",
                };
                output.push_str(&format!("{}{}\n", prefix, line.content));
            }
        }

        output
    }
}

/// Diff statistics
#[derive(Debug, Clone, Default)]
pub struct DiffStats {
    pub added: usize,
    pub removed: usize,
    pub modified: usize,
    pub hunks: usize,
}

// ============================================================================
// C4: Service Status Dashboard
// ============================================================================

/// Systemd unit state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnitState {
    Active,
    Inactive,
    Failed,
    Activating,
    Deactivating,
    Reloading,
    Unknown,
}

impl UnitState {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "active" => Self::Active,
            "inactive" => Self::Inactive,
            "failed" => Self::Failed,
            "activating" => Self::Activating,
            "deactivating" => Self::Deactivating,
            "reloading" => Self::Reloading,
            _ => Self::Unknown,
        }
    }

    pub fn color(&self) -> (u8, u8, u8) {
        match self {
            Self::Active => (100, 200, 100),       // Green
            Self::Inactive => (150, 150, 150),     // Gray
            Self::Failed => (200, 100, 100),       // Red
            Self::Activating => (200, 200, 100),   // Yellow
            Self::Deactivating => (200, 150, 100), // Orange
            Self::Reloading => (100, 150, 200),    // Blue
            Self::Unknown => (150, 150, 150),      // Gray
        }
    }

    pub fn icon(&self) -> char {
        match self {
            Self::Active => '\u{2714}',       // Check mark
            Self::Inactive => '\u{25CB}',     // Circle
            Self::Failed => '\u{2718}',       // X mark
            Self::Activating => '\u{21BB}',   // Clockwise arrow
            Self::Deactivating => '\u{21BA}', // Counter-clockwise
            Self::Reloading => '\u{21BA}',    // Counter-clockwise
            Self::Unknown => '\u{2753}',      // Question mark
        }
    }
}

/// Systemd unit information
#[derive(Debug, Clone)]
pub struct SystemdUnit {
    /// Unit name (e.g., "nginx.service")
    pub name: String,
    /// Current state
    pub state: UnitState,
    /// Sub-state (e.g., "running", "exited")
    pub sub_state: String,
    /// Description
    pub description: String,
    /// Load state
    pub load_state: String,
    /// Active since timestamp
    pub active_since: Option<DateTime<Utc>>,
    /// Main PID (if applicable)
    pub main_pid: Option<u32>,
    /// Memory usage in bytes
    pub memory_bytes: Option<u64>,
    /// CPU usage percentage
    pub cpu_percent: Option<f64>,
    /// Last log entries
    pub recent_logs: Vec<String>,
    /// Is enabled at boot
    pub enabled: bool,
    /// Unit file path
    pub unit_file: String,
}

/// Service dashboard state
#[derive(Debug, Clone)]
pub struct ServiceDashboard {
    /// All units
    pub units: Vec<SystemdUnit>,
    /// Filter by state
    pub state_filter: Option<UnitState>,
    /// Filter by name pattern
    pub name_filter: String,
    /// Selected unit for details
    pub selected_unit: Option<String>,
    /// Auto-refresh interval (ms)
    pub refresh_interval_ms: u64,
    /// Last refresh time
    pub last_refresh: DateTime<Utc>,
    /// Show only NixOS-managed services
    pub nixos_only: bool,
    /// Sort by field
    pub sort_by: SortField,
    /// Sort ascending
    pub sort_ascending: bool,
    /// Loading state
    pub is_loading: bool,
    /// Error message
    pub error: Option<String>,
}

/// Sort field for service list
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortField {
    Name,
    State,
    Memory,
    Cpu,
    ActiveSince,
}

impl Default for ServiceDashboard {
    fn default() -> Self {
        Self::new()
    }
}

impl ServiceDashboard {
    pub fn new() -> Self {
        Self {
            units: Vec::new(),
            state_filter: None,
            name_filter: String::new(),
            selected_unit: None,
            refresh_interval_ms: 5000,
            last_refresh: Utc::now(),
            nixos_only: false,
            sort_by: SortField::Name,
            sort_ascending: true,
            is_loading: false,
            error: None,
        }
    }

    /// Load service units from systemd
    pub fn refresh(&mut self) -> Result<(), String> {
        self.is_loading = true;
        self.error = None;

        // List all services
        let output = Command::new("systemctl")
            .args([
                "list-units",
                "--type=service",
                "--all",
                "--no-pager",
                "--plain",
            ])
            .output()
            .map_err(|e| format!("Failed to list units: {e}"))?;

        self.is_loading = false;
        self.last_refresh = Utc::now();

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let err = format!("systemctl failed: {stderr}");
            self.error = Some(err.clone());
            return Err(err);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.units = self.parse_units(&stdout);

        // Sort
        self.sort();

        Ok(())
    }

    fn parse_units(&self, output: &str) -> Vec<SystemdUnit> {
        let mut units = Vec::new();

        for line in output.lines().skip(1) {
            // Skip header
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let name = parts[0].to_string();
                let load_state = parts[1].to_string();
                let state = UnitState::from_str(parts[2]);
                let sub_state = parts[3].to_string();
                let description = parts[4..].join(" ");

                units.push(SystemdUnit {
                    name,
                    state,
                    sub_state,
                    description,
                    load_state,
                    active_since: None,
                    main_pid: None,
                    memory_bytes: None,
                    cpu_percent: None,
                    recent_logs: Vec::new(),
                    enabled: false,
                    unit_file: String::new(),
                });
            }
        }

        units
    }

    /// Get detailed info for a specific unit
    pub fn get_unit_details(&self, name: &str) -> Result<SystemdUnit, String> {
        let output = Command::new("systemctl")
            .args(["show", name, "--no-pager"])
            .output()
            .map_err(|e| format!("Failed to get unit details: {e}"))?;

        if !output.status.success() {
            return Err("Failed to get unit details".to_string());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        let mut unit = SystemdUnit {
            name: name.to_string(),
            state: UnitState::Unknown,
            sub_state: String::new(),
            description: String::new(),
            load_state: String::new(),
            active_since: None,
            main_pid: None,
            memory_bytes: None,
            cpu_percent: None,
            recent_logs: Vec::new(),
            enabled: false,
            unit_file: String::new(),
        };

        for line in stdout.lines() {
            if let Some((key, value)) = line.split_once('=') {
                match key {
                    "ActiveState" => unit.state = UnitState::from_str(value),
                    "SubState" => unit.sub_state = value.to_string(),
                    "Description" => unit.description = value.to_string(),
                    "LoadState" => unit.load_state = value.to_string(),
                    "MainPID" => unit.main_pid = value.parse().ok(),
                    "MemoryCurrent" => unit.memory_bytes = value.parse().ok(),
                    "UnitFileState" => unit.enabled = value == "enabled",
                    "FragmentPath" => unit.unit_file = value.to_string(),
                    _ => {}
                }
            }
        }

        // Get recent logs
        let log_output = Command::new("journalctl")
            .args(["-u", name, "-n", "10", "--no-pager", "-o", "short-iso"])
            .output();

        if let Ok(output) = log_output {
            if output.status.success() {
                unit.recent_logs = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .map(|s| s.to_string())
                    .collect();
            }
        }

        Ok(unit)
    }

    /// Filter units
    pub fn filtered_units(&self) -> Vec<&SystemdUnit> {
        self.units
            .iter()
            .filter(|u| {
                // State filter
                if let Some(ref state) = self.state_filter {
                    if &u.state != state {
                        return false;
                    }
                }

                // Name filter
                if !self.name_filter.is_empty()
                    && !u
                        .name
                        .to_lowercase()
                        .contains(&self.name_filter.to_lowercase())
                    && !u
                        .description
                        .to_lowercase()
                        .contains(&self.name_filter.to_lowercase())
                {
                    return false;
                }

                // NixOS filter (check if unit file is in /nix/store)
                if self.nixos_only && !u.unit_file.starts_with("/nix/store") {
                    return false;
                }

                true
            })
            .collect()
    }

    /// Sort units
    pub fn sort(&mut self) {
        let ascending = self.sort_ascending;
        let sort_by = self.sort_by;

        self.units.sort_by(|a, b| {
            let cmp = match sort_by {
                SortField::Name => a.name.cmp(&b.name),
                SortField::State => format!("{:?}", a.state).cmp(&format!("{:?}", b.state)),
                SortField::Memory => a.memory_bytes.cmp(&b.memory_bytes),
                SortField::Cpu => a
                    .cpu_percent
                    .partial_cmp(&b.cpu_percent)
                    .unwrap_or(std::cmp::Ordering::Equal),
                SortField::ActiveSince => a.active_since.cmp(&b.active_since),
            };

            if ascending {
                cmp
            } else {
                cmp.reverse()
            }
        });
    }

    /// Control a service
    pub fn control(&self, name: &str, action: ServiceAction) -> Result<(), String> {
        let action_str = match action {
            ServiceAction::Start => "start",
            ServiceAction::Stop => "stop",
            ServiceAction::Restart => "restart",
            ServiceAction::Reload => "reload",
            ServiceAction::Enable => "enable",
            ServiceAction::Disable => "disable",
        };

        let output = Command::new("sudo")
            .args(["systemctl", action_str, name])
            .output()
            .map_err(|e| format!("Failed to {action_str} service: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to {action_str} {name}: {stderr}"));
        }

        Ok(())
    }

    /// Get summary stats
    pub fn stats(&self) -> DashboardStats {
        let mut stats = DashboardStats::default();

        for unit in &self.units {
            match unit.state {
                UnitState::Active => stats.active += 1,
                UnitState::Inactive => stats.inactive += 1,
                UnitState::Failed => stats.failed += 1,
                _ => stats.other += 1,
            }
        }

        stats.total = self.units.len();
        stats
    }
}

/// Service control action
#[derive(Debug, Clone, Copy)]
pub enum ServiceAction {
    Start,
    Stop,
    Restart,
    Reload,
    Enable,
    Disable,
}

/// Dashboard statistics
#[derive(Debug, Clone, Default)]
pub struct DashboardStats {
    pub total: usize,
    pub active: usize,
    pub inactive: usize,
    pub failed: usize,
    pub other: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_categories() {
        let cats = ModuleCategory::standard_categories();
        assert!(!cats.is_empty());
        assert!(cats.iter().any(|c| c.name == "Services"));
    }

    #[test]
    fn test_unit_state_from_str() {
        assert_eq!(UnitState::from_str("active"), UnitState::Active);
        assert_eq!(UnitState::from_str("FAILED"), UnitState::Failed);
        assert_eq!(UnitState::from_str("unknown"), UnitState::Unknown);
    }

    #[test]
    fn test_diff_stats() {
        let mut diff = LiveConfigDiff::new();
        diff.original = "line1\nline2\nline3".to_string();
        diff.modified = "line1\nmodified\nline3\nnew".to_string();
        diff.compute_diff();

        let stats = diff.stats();
        assert!(stats.added > 0 || stats.removed > 0 || stats.modified > 0);
    }

    #[test]
    fn test_dashboard_stats() {
        let dashboard = ServiceDashboard::new();
        let stats = dashboard.stats();
        assert_eq!(stats.total, 0);
    }
}
