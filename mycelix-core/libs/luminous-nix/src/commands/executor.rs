// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Command executor for NixOS operations
//!
//! Executes Nix commands asynchronously with JSON output parsing
//! for 10x faster results compared to text parsing.

use anyhow::{Context, Result};
use std::process::Stdio;
use std::time::Instant;
use tokio::process::Command;

use super::types::*;

/// Executor for NixOS commands
pub struct CommandExecutor {
    /// Use flakes mode
    use_flakes: bool,
}

impl CommandExecutor {
    /// Create a new command executor
    pub fn new() -> Self {
        Self { use_flakes: true }
    }

    /// Create executor with flakes disabled
    pub fn without_flakes() -> Self {
        Self { use_flakes: false }
    }

    /// Search for packages
    ///
    /// Uses `nix search nixpkgs <query> --json` for fast results
    pub async fn search(&self, query: &str, limit: usize, dry_run: bool) -> Result<SearchResult> {
        let _cmd = format!("nix search nixpkgs {} --json", query);

        if dry_run {
            return Ok(SearchResult {
                packages: vec![],
                total_count: 0,
                query: query.to_string(),
                dry_run: true,
            });
        }

        let start = Instant::now();
        let output = Command::new("nix")
            .args(["search", "nixpkgs", query, "--json"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .context("Failed to execute nix search")?;

        let _duration = start.elapsed().as_millis() as u64;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Check if it's just "no results" vs actual error
            if stderr.contains("error:") && !stderr.contains("no results") {
                return Err(anyhow::anyhow!("Search failed: {}", stderr));
            }
            // No results is not an error
            return Ok(SearchResult {
                packages: vec![],
                total_count: 0,
                query: query.to_string(),
                dry_run: false,
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse JSON output
        let packages = self.parse_search_json(&stdout, limit)?;
        let total_count = packages.len();

        Ok(SearchResult {
            packages,
            total_count,
            query: query.to_string(),
            dry_run: false,
        })
    }

    /// Parse JSON output from nix search
    fn parse_search_json(&self, json_str: &str, limit: usize) -> Result<Vec<PackageInfo>> {
        if json_str.trim().is_empty() {
            return Ok(vec![]);
        }

        let json: serde_json::Value =
            serde_json::from_str(json_str).context("Failed to parse nix search JSON")?;

        let map = json.as_object().context("Expected JSON object")?;

        let mut packages: Vec<PackageInfo> = map
            .iter()
            .filter_map(|(attr_path, info)| PackageInfo::from_nix_json(attr_path, info))
            .take(limit)
            .collect();

        // Sort by name
        packages.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(packages)
    }

    /// Install a package
    pub async fn install(
        &self,
        package: &str,
        use_flake: bool,
        dry_run: bool,
    ) -> Result<ExecutionResult> {
        let cmd = if use_flake || self.use_flakes {
            format!("nix profile install nixpkgs#{}", package)
        } else {
            format!("nix-env -iA nixpkgs.{}", package)
        };

        if dry_run {
            return Ok(ExecutionResult::dry_run(cmd));
        }

        let start = Instant::now();

        let output = if use_flake || self.use_flakes {
            Command::new("nix")
                .args(["profile", "install", &format!("nixpkgs#{}", package)])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await
        } else {
            Command::new("nix-env")
                .args(["-iA", &format!("nixpkgs.{}", package)])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await
        };

        let output = output.context("Failed to execute install command")?;
        let duration = start.elapsed().as_millis() as u64;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            Ok(ExecutionResult::success(cmd, stdout, duration))
        } else {
            Ok(ExecutionResult::failure(
                cmd,
                stderr,
                output.status.code().unwrap_or(-1),
                duration,
            ))
        }
    }

    /// Remove a package
    pub async fn remove(&self, package: &str, dry_run: bool) -> Result<ExecutionResult> {
        let cmd = if self.use_flakes {
            format!("nix profile remove {}", package)
        } else {
            format!("nix-env -e {}", package)
        };

        if dry_run {
            return Ok(ExecutionResult::dry_run(cmd));
        }

        let start = Instant::now();

        let output = if self.use_flakes {
            Command::new("nix")
                .args(["profile", "remove", package])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await
        } else {
            Command::new("nix-env")
                .args(["-e", package])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await
        };

        let output = output.context("Failed to execute remove command")?;
        let duration = start.elapsed().as_millis() as u64;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            Ok(ExecutionResult::success(cmd, stdout, duration))
        } else {
            Ok(ExecutionResult::failure(
                cmd,
                stderr,
                output.status.code().unwrap_or(-1),
                duration,
            ))
        }
    }

    /// List system generations
    pub async fn list_generations(
        &self,
        limit: Option<usize>,
        dry_run: bool,
    ) -> Result<GenerationList> {
        let _cmd = "nixos-rebuild list-generations".to_string();

        if dry_run {
            return Ok(GenerationList {
                generations: vec![],
                current: 0,
                dry_run: true,
            });
        }

        let output = Command::new("nixos-rebuild")
            .args(["list-generations"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .context("Failed to list generations")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let generations = self.parse_generations(&stdout, limit)?;

        let current = generations
            .iter()
            .find(|g| g.current)
            .map(|g| g.number)
            .unwrap_or(0);

        Ok(GenerationList {
            generations,
            current,
            dry_run: false,
        })
    }

    /// Parse generation output
    fn parse_generations(&self, output: &str, limit: Option<usize>) -> Result<Vec<Generation>> {
        let mut generations = Vec::new();

        for line in output.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Parse lines like:
            // "42  2024-01-15 14:30:00  (current)"
            // "41  2024-01-14 10:20:00"
            let current = line.contains("(current)");
            let line = line.replace("(current)", "").trim().to_string();

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(number) = parts[0].parse::<u32>() {
                    let date = format!("{} {}", parts[1], parts.get(2).unwrap_or(&""));
                    generations.push(Generation {
                        number,
                        date,
                        nixos_version: None,
                        kernel_version: None,
                        config_revision: None,
                        current,
                    });
                }
            }
        }

        // Sort by generation number (descending)
        generations.sort_by(|a, b| b.number.cmp(&a.number));

        // Apply limit
        if let Some(limit) = limit {
            generations.truncate(limit);
        }

        Ok(generations)
    }

    /// Get system status
    pub async fn system_status(&self, dry_run: bool) -> Result<SystemStatus> {
        if dry_run {
            return Ok(SystemStatus {
                nixos_version: "dry-run".to_string(),
                kernel_version: "dry-run".to_string(),
                current_generation: 0,
                total_generations: 0,
                store_size: None,
                last_rebuild: None,
            });
        }

        // Get NixOS version
        let nixos_version = self.get_nixos_version().await.unwrap_or_default();

        // Get kernel version
        let kernel_version = self.get_kernel_version().await.unwrap_or_default();

        // Get generation info
        let generations = self.list_generations(None, false).await?;
        let current_generation = generations.current;
        let total_generations = generations.generations.len() as u32;

        // Get store size
        let store_size = self.get_store_size().await.ok();

        Ok(SystemStatus {
            nixos_version,
            kernel_version,
            current_generation,
            total_generations,
            store_size,
            last_rebuild: None,
        })
    }

    async fn get_nixos_version(&self) -> Result<String> {
        let output = Command::new("nixos-version")
            .stdout(Stdio::piped())
            .output()
            .await?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    async fn get_kernel_version(&self) -> Result<String> {
        let output = Command::new("uname")
            .args(["-r"])
            .stdout(Stdio::piped())
            .output()
            .await?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    async fn get_store_size(&self) -> Result<String> {
        let output = Command::new("du")
            .args(["-sh", "/nix/store"])
            .stdout(Stdio::piped())
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.split_whitespace().next().unwrap_or("unknown").to_string())
    }

    /// Garbage collect
    pub async fn garbage_collect(
        &self,
        older_than: Option<u32>,
        dry_run: bool,
    ) -> Result<ExecutionResult> {
        let cmd = match older_than {
            Some(days) => format!("nix-collect-garbage --delete-older-than {}d", days),
            None => "nix-collect-garbage -d".to_string(),
        };

        if dry_run {
            return Ok(ExecutionResult::dry_run(cmd));
        }

        let start = Instant::now();

        let mut command = Command::new("nix-collect-garbage");
        match older_than {
            Some(days) => {
                command.args(["--delete-older-than", &format!("{}d", days)]);
            }
            None => {
                command.arg("-d");
            }
        }

        let output = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .context("Failed to run garbage collection")?;

        let duration = start.elapsed().as_millis() as u64;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            Ok(ExecutionResult::success(cmd, stdout, duration))
        } else {
            Ok(ExecutionResult::failure(
                cmd,
                stderr,
                output.status.code().unwrap_or(-1),
                duration,
            ))
        }
    }

    /// Rebuild the system
    pub async fn rebuild(&self, switch: bool, dry_run: bool) -> Result<ExecutionResult> {
        let action = if switch { "switch" } else { "build" };
        let cmd = format!("sudo nixos-rebuild {}", action);

        if dry_run {
            return Ok(ExecutionResult::dry_run(cmd));
        }

        let start = Instant::now();

        let output = Command::new("sudo")
            .args(["nixos-rebuild", action])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .context("Failed to rebuild system")?;

        let duration = start.elapsed().as_millis() as u64;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            Ok(ExecutionResult::success(cmd, stdout, duration))
        } else {
            Ok(ExecutionResult::failure(
                cmd,
                stderr,
                output.status.code().unwrap_or(-1),
                duration,
            ))
        }
    }
}

impl Default for CommandExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_search_json() {
        let executor = CommandExecutor::new();
        let json = r#"
        {
            "legacyPackages.x86_64-linux.firefox": {
                "pname": "firefox",
                "version": "120.0",
                "description": "A web browser"
            },
            "legacyPackages.x86_64-linux.firefox-esr": {
                "pname": "firefox-esr",
                "version": "115.6.0",
                "description": "A web browser (ESR)"
            }
        }
        "#;

        let packages = executor.parse_search_json(json, 10).unwrap();
        assert_eq!(packages.len(), 2);
        assert!(packages.iter().any(|p| p.name == "firefox"));
    }

    #[test]
    fn test_dry_run() {
        let executor = CommandExecutor::new();
        let result = ExecutionResult::dry_run("test command".to_string());
        assert!(result.dry_run);
        assert!(result.success);
    }
}
