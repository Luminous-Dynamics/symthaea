// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! I2: Flake Input Updater
//!
//! Update flake inputs with diff preview and selective updates.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, SystemTime};

/// Flake input information
#[derive(Debug, Clone)]
pub struct FlakeInput {
    /// Input name
    pub name: String,
    /// Input URL/path
    pub url: String,
    /// Current locked revision
    pub locked_rev: Option<String>,
    /// Current locked date
    pub locked_date: Option<String>,
    /// Last checked for updates
    pub last_checked: Option<SystemTime>,
    /// Whether update is available
    pub update_available: bool,
    /// New revision if update available
    pub new_rev: Option<String>,
}

/// Update preview showing what would change
#[derive(Debug, Clone)]
pub struct UpdatePreview {
    /// Input being updated
    pub input: String,
    /// Current revision (short)
    pub from_rev: String,
    /// New revision (short)
    pub to_rev: String,
    /// Commit messages between versions
    pub commits: Vec<String>,
    /// Breaking changes detected
    pub breaking_changes: Vec<String>,
    /// Estimated impact
    pub impact: UpdateImpact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateImpact {
    /// No impact expected
    None,
    /// Minor update (patch)
    Minor,
    /// Moderate update (minor version)
    Moderate,
    /// Major update (potentially breaking)
    Major,
    /// Unknown impact
    Unknown,
}

impl UpdateImpact {
    pub fn icon(&self) -> &'static str {
        match self {
            UpdateImpact::None => "✓",
            UpdateImpact::Minor => "↑",
            UpdateImpact::Moderate => "⬆",
            UpdateImpact::Major => "⚠",
            UpdateImpact::Unknown => "?",
        }
    }
}

/// Result of an update operation
#[derive(Debug, Clone)]
pub struct UpdateResult {
    /// Input name
    pub input: String,
    /// Success
    pub success: bool,
    /// Old revision
    pub old_rev: Option<String>,
    /// New revision
    pub new_rev: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
    /// Time taken
    pub duration: Duration,
}

/// Flake input updater
pub struct FlakeUpdater {
    /// Flake directory
    flake_dir: PathBuf,
    /// Cached inputs
    inputs: HashMap<String, FlakeInput>,
    /// Update history
    history: Vec<UpdateResult>,
}

impl FlakeUpdater {
    /// Create new updater for a flake directory
    pub fn new(flake_dir: impl Into<PathBuf>) -> Self {
        Self {
            flake_dir: flake_dir.into(),
            inputs: HashMap::new(),
            history: Vec::new(),
        }
    }

    /// Refresh list of flake inputs
    pub fn refresh_inputs(&mut self) -> Result<Vec<FlakeInput>, FlakeError> {
        self.inputs.clear();

        // Read flake.lock
        let lock_path = self.flake_dir.join("flake.lock");
        if !lock_path.exists() {
            return Err(FlakeError::NoLockFile);
        }

        let lock_content =
            std::fs::read_to_string(&lock_path).map_err(|e| FlakeError::IoError(e.to_string()))?;

        // Parse flake.lock (JSON format)
        let lock: serde_json::Value = serde_json::from_str(&lock_content)
            .map_err(|e| FlakeError::ParseError(e.to_string()))?;

        if let Some(nodes) = lock.get("nodes").and_then(|n| n.as_object()) {
            for (name, node) in nodes {
                if name == "root" {
                    continue;
                }

                let locked = node.get("locked");
                let original = node.get("original");

                let url = original
                    .and_then(|o| o.get("url"))
                    .or_else(|| original.and_then(|o| o.get("owner")).and(original))
                    .map(|v| {
                        if let Some(owner) = original
                            .and_then(|o| o.get("owner"))
                            .and_then(|o| o.as_str())
                        {
                            let repo = original
                                .and_then(|o| o.get("repo"))
                                .and_then(|r| r.as_str())
                                .unwrap_or("");
                            format!("github:{owner}/{repo}")
                        } else {
                            v.as_str().unwrap_or("").to_string()
                        }
                    })
                    .unwrap_or_default();

                let rev = locked
                    .and_then(|l| l.get("rev"))
                    .and_then(|r| r.as_str())
                    .map(|s| s.to_string());

                let date = locked
                    .and_then(|l| l.get("lastModified"))
                    .and_then(|d| d.as_u64())
                    .map(|ts| {
                        // Format timestamp
                        let duration = std::time::Duration::from_secs(ts);
                        let datetime = std::time::UNIX_EPOCH + duration;
                        format!("{datetime:?}")
                    });

                let input = FlakeInput {
                    name: name.clone(),
                    url,
                    locked_rev: rev,
                    locked_date: date,
                    last_checked: None,
                    update_available: false,
                    new_rev: None,
                };

                self.inputs.insert(name.clone(), input);
            }
        }

        Ok(self.inputs.values().cloned().collect())
    }

    /// Check for updates on a specific input
    pub fn check_updates(&mut self, input_name: &str) -> Result<bool, FlakeError> {
        let output = Command::new("nix")
            .args(["flake", "update", input_name, "--dry-run"])
            .current_dir(&self.flake_dir)
            .output()
            .map_err(|e| FlakeError::CommandFailed(e.to_string()))?;

        let stderr = String::from_utf8_lossy(&output.stderr);

        // Check if update would change anything
        let update_available = stderr.contains("updating") || stderr.contains("→");

        if let Some(input) = self.inputs.get_mut(input_name) {
            input.last_checked = Some(SystemTime::now());
            input.update_available = update_available;

            // Try to extract new revision from output
            if update_available {
                // Look for "xxx → yyy" pattern
                for line in stderr.lines() {
                    if line.contains("→") {
                        let parts: Vec<&str> = line.split("→").collect();
                        if parts.len() >= 2 {
                            let new_rev = parts[1].trim().chars().take(12).collect();
                            input.new_rev = Some(new_rev);
                        }
                    }
                }
            }
        }

        Ok(update_available)
    }

    /// Check all inputs for updates
    pub fn check_all_updates(&mut self) -> Result<Vec<String>, FlakeError> {
        let input_names: Vec<String> = self.inputs.keys().cloned().collect();
        let mut updates = Vec::new();

        for name in input_names {
            if self.check_updates(&name)? {
                updates.push(name);
            }
        }

        Ok(updates)
    }

    /// Get update preview for an input
    pub fn preview_update(&self, input_name: &str) -> Result<UpdatePreview, FlakeError> {
        let input = self
            .inputs
            .get(input_name)
            .ok_or_else(|| FlakeError::InputNotFound(input_name.to_string()))?;

        let from_rev = input
            .locked_rev
            .clone()
            .map(|r| r.chars().take(7).collect())
            .unwrap_or_else(|| "unknown".to_string());

        let to_rev = input
            .new_rev
            .clone()
            .unwrap_or_else(|| "latest".to_string());

        // Estimate impact based on rev difference
        let impact = if input.new_rev.is_none() {
            UpdateImpact::Unknown
        } else if from_rev == to_rev {
            UpdateImpact::None
        } else {
            // Simple heuristic - could be improved with actual diff analysis
            UpdateImpact::Moderate
        };

        Ok(UpdatePreview {
            input: input_name.to_string(),
            from_rev,
            to_rev,
            commits: Vec::new(), // Would need GitHub API to fetch
            breaking_changes: Vec::new(),
            impact,
        })
    }

    /// Update a specific input
    pub fn update_input(&mut self, input_name: &str) -> Result<UpdateResult, FlakeError> {
        let start = std::time::Instant::now();

        let old_rev = self
            .inputs
            .get(input_name)
            .and_then(|i| i.locked_rev.clone());

        let output = Command::new("nix")
            .args(["flake", "update", input_name])
            .current_dir(&self.flake_dir)
            .output()
            .map_err(|e| FlakeError::CommandFailed(e.to_string()))?;

        let duration = start.elapsed();

        let result = if output.status.success() {
            // Refresh to get new revision
            self.refresh_inputs()?;

            let new_rev = self
                .inputs
                .get(input_name)
                .and_then(|i| i.locked_rev.clone());

            UpdateResult {
                input: input_name.to_string(),
                success: true,
                old_rev,
                new_rev,
                error: None,
                duration,
            }
        } else {
            UpdateResult {
                input: input_name.to_string(),
                success: false,
                old_rev,
                new_rev: None,
                error: Some(String::from_utf8_lossy(&output.stderr).to_string()),
                duration,
            }
        };

        self.history.push(result.clone());
        Ok(result)
    }

    /// Update all inputs
    pub fn update_all(&mut self) -> Result<Vec<UpdateResult>, FlakeError> {
        let start = std::time::Instant::now();

        // Store old revisions
        let old_revs: HashMap<String, Option<String>> = self
            .inputs
            .iter()
            .map(|(k, v)| (k.clone(), v.locked_rev.clone()))
            .collect();

        let output = Command::new("nix")
            .args(["flake", "update"])
            .current_dir(&self.flake_dir)
            .output()
            .map_err(|e| FlakeError::CommandFailed(e.to_string()))?;

        let duration = start.elapsed();

        if output.status.success() {
            // Refresh to get new revisions
            self.refresh_inputs()?;

            let results: Vec<UpdateResult> = self
                .inputs
                .iter()
                .map(|(name, input)| {
                    let old_rev = old_revs.get(name).cloned().flatten();
                    let new_rev = input.locked_rev.clone();
                    let _changed = old_rev != new_rev;

                    UpdateResult {
                        input: name.clone(),
                        success: true,
                        old_rev,
                        new_rev,
                        error: None,
                        duration: duration / self.inputs.len() as u32,
                    }
                })
                .collect();

            for result in &results {
                self.history.push(result.clone());
            }

            Ok(results)
        } else {
            Err(FlakeError::CommandFailed(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ))
        }
    }

    /// Get input by name
    pub fn get_input(&self, name: &str) -> Option<&FlakeInput> {
        self.inputs.get(name)
    }

    /// Get all inputs
    pub fn inputs(&self) -> Vec<&FlakeInput> {
        self.inputs.values().collect()
    }

    /// Get update history
    pub fn history(&self) -> &[UpdateResult] {
        &self.history
    }

    /// Format inputs for display
    pub fn format_inputs(&self) -> String {
        let mut output = String::new();
        let mut inputs: Vec<_> = self.inputs.values().collect();
        inputs.sort_by(|a, b| a.name.cmp(&b.name));

        for input in inputs {
            let rev = input
                .locked_rev
                .as_deref()
                .map(|r| &r[..7.min(r.len())])
                .unwrap_or("???");

            let status = if input.update_available {
                "🔄"
            } else if input.last_checked.is_some() {
                "✓"
            } else {
                " "
            };

            output.push_str(&format!(
                "{} {} ({}) - {}\n",
                status, input.name, rev, input.url
            ));
        }

        output
    }
}

/// Flake operation errors
#[derive(Debug, Clone)]
pub enum FlakeError {
    /// No flake.lock file found
    NoLockFile,
    /// Input not found
    InputNotFound(String),
    /// IO error
    IoError(String),
    /// Parse error
    ParseError(String),
    /// Command failed
    CommandFailed(String),
}

impl std::fmt::Display for FlakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoLockFile => write!(f, "No flake.lock file found"),
            Self::InputNotFound(n) => write!(f, "Input not found: {n}"),
            Self::IoError(e) => write!(f, "IO error: {e}"),
            Self::ParseError(e) => write!(f, "Parse error: {e}"),
            Self::CommandFailed(e) => write!(f, "Command failed: {e}"),
        }
    }
}

impl std::error::Error for FlakeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_impact() {
        assert_eq!(UpdateImpact::None.icon(), "✓");
        assert_eq!(UpdateImpact::Major.icon(), "⚠");
    }

    #[test]
    fn test_flake_updater_creation() {
        let updater = FlakeUpdater::new("/etc/nixos");
        assert!(updater.inputs.is_empty());
    }
}
