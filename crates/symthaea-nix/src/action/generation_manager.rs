// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS Generation Management
//!
//! Manages NixOS generations: list, switch, rollback, delete, and diff.
//! Produces `NixOSCommand` values for anything that modifies state;
//! read-only queries (list, diff) run directly.

use super::executor::{NixOSCommand, SafetyLevel};
use std::process::Command;

/// Manages NixOS generations: switch, rollback, delete, and boot configuration.
pub struct GenerationManager;

/// A single NixOS generation entry.
#[derive(Debug, Clone)]
pub struct Generation {
    /// Generation number.
    pub number: u32,
    /// Date string from nixos-rebuild list-generations.
    pub date: String,
    /// NixOS version string.
    pub nixos_version: String,
    /// Kernel version string.
    pub kernel_version: String,
    /// Whether this is the current generation.
    pub current: bool,
}

/// Diff between two generations.
#[derive(Debug, Clone)]
pub struct GenerationDiff {
    /// Generation numbers being compared.
    pub from: u32,
    pub to: u32,
    /// Packages added in the newer generation.
    pub added: Vec<String>,
    /// Packages removed in the newer generation.
    pub removed: Vec<String>,
    /// Packages changed (version bump etc).
    pub changed: Vec<(String, String, String)>, // (name, old_version, new_version)
}

impl GenerationManager {
    /// List all NixOS generations by parsing `nixos-rebuild list-generations`.
    pub fn list() -> Result<Vec<Generation>, std::io::Error> {
        let output = Command::new("nixos-rebuild")
            .args(["list-generations"])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "nixos-rebuild list-generations failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_generations(&stdout)
    }

    /// Get the current generation number.
    pub fn current_generation() -> Result<u32, std::io::Error> {
        let gens = Self::list()?;
        gens.iter()
            .find(|g| g.current)
            .map(|g| g.number)
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "No current generation found")
            })
    }

    /// Generate a command to switch to a specific generation.
    pub fn switch_to(generation: u32) -> NixOSCommand {
        NixOSCommand::Custom {
            command: "nix-env".to_string(),
            args: vec![
                "--switch-generation".to_string(),
                generation.to_string(),
                "-p".to_string(),
                "/nix/var/nix/profiles/system".to_string(),
            ],
            safety_level: SafetyLevel::SystemCritical,
        }
    }

    /// Generate a rollback command (switch to previous generation).
    pub fn rollback() -> NixOSCommand {
        NixOSCommand::Custom {
            command: "nixos-rebuild".to_string(),
            args: vec!["switch".to_string(), "--rollback".to_string()],
            safety_level: SafetyLevel::SystemCritical,
        }
    }

    /// Generate a command to delete old generations.
    ///
    /// `keep_last` specifies how many recent generations to keep.
    pub fn delete_old(keep_last: usize) -> NixOSCommand {
        // nix-env --delete-generations +N keeps last N generations
        NixOSCommand::Custom {
            command: "nix-env".to_string(),
            args: vec![
                "--delete-generations".to_string(),
                format!("+{}", keep_last),
                "-p".to_string(),
                "/nix/var/nix/profiles/system".to_string(),
            ],
            safety_level: SafetyLevel::Destructive,
        }
    }

    /// Generate a command to delete generations older than N days.
    pub fn delete_older_than(days: u32) -> NixOSCommand {
        NixOSCommand::Custom {
            command: "nix-env".to_string(),
            args: vec![
                "--delete-generations".to_string(),
                format!("{}d", days),
                "-p".to_string(),
                "/nix/var/nix/profiles/system".to_string(),
            ],
            safety_level: SafetyLevel::Destructive,
        }
    }

    /// Diff two generations by comparing their store closures.
    ///
    /// Uses `nix store diff-closures` to compare the store paths.
    pub fn diff(from: u32, to: u32) -> Result<GenerationDiff, std::io::Error> {
        let from_path = format!("/nix/var/nix/profiles/system-{from}-link");
        let to_path = format!("/nix/var/nix/profiles/system-{to}-link");

        let output = Command::new("nix")
            .args(["store", "diff-closures", &from_path, &to_path])
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_diff(&stdout, from, to)
    }

    /// Parse the output of `nixos-rebuild list-generations`.
    ///
    /// Format varies, but typically:
    /// ```text
    ///  42  2025-01-15  NixOS 24.11  linux 6.12.3  (current)
    ///  41  2025-01-10  NixOS 24.11  linux 6.12.3
    /// ```
    fn parse_generations(output: &str) -> Result<Vec<Generation>, std::io::Error> {
        let mut generations = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let current = trimmed.contains("(current)");
            let clean = trimmed.replace("(current)", "").trim().to_string();
            let tokens: Vec<&str> = clean.split_whitespace().collect();

            if tokens.is_empty() {
                continue;
            }

            let number = match tokens[0].parse::<u32>() {
                Ok(n) => n,
                Err(_) => continue,
            };

            let date = tokens.get(1).copied().unwrap_or("").to_string();

            // NixOS version is usually tokens 2..3
            let nixos_version = if tokens.len() > 3 {
                format!("{} {}", tokens[2], tokens[3])
            } else if tokens.len() > 2 {
                tokens[2].to_string()
            } else {
                String::new()
            };

            // Kernel version is usually tokens 4..5
            let kernel_version = if tokens.len() > 5 {
                format!("{} {}", tokens[4], tokens[5])
            } else if tokens.len() > 4 {
                tokens[4].to_string()
            } else {
                String::new()
            };

            generations.push(Generation {
                number,
                date,
                nixos_version,
                kernel_version,
                current,
            });
        }

        Ok(generations)
    }

    /// Parse the output of `nix store diff-closures`.
    ///
    /// Format:
    /// ```text
    /// firefox: 120.0 → 121.0, +12.3 MiB
    /// vim: ∅ → 9.0.2100, +45.2 MiB
    /// old-pkg: 1.0 → ∅, -5.1 MiB
    /// ```
    fn parse_diff(output: &str, from: u32, to: u32) -> Result<GenerationDiff, std::io::Error> {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut changed = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some((name, rest)) = trimmed.split_once(':') {
                let name = name.trim().to_string();
                let rest = rest.trim();

                if rest.contains("∅ →") || rest.contains("∅ ->") {
                    // New package
                    added.push(name);
                } else if rest.contains("→ ∅") || rest.contains("-> ∅") {
                    // Removed package
                    removed.push(name);
                } else if rest.contains('→') || rest.contains("->") {
                    // Changed package
                    let arrow = if rest.contains('→') { '→' } else { '-' };
                    let parts: Vec<&str> = rest.splitn(2, arrow).collect();
                    if parts.len() == 2 {
                        let old_ver = parts[0].trim().to_string();
                        let new_ver = parts[1]
                            .trim()
                            .split(',')
                            .next()
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        changed.push((name, old_ver, new_ver));
                    }
                }
            }
        }

        Ok(GenerationDiff {
            from,
            to,
            added,
            removed,
            changed,
        })
    }

    /// Count total generations.
    pub fn generation_count() -> Result<usize, std::io::Error> {
        Self::list().map(|g| g.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOCK_GENERATIONS: &str = "\
  45  2025-01-20  NixOS 24.11  linux 6.12.3  (current)
  44  2025-01-15  NixOS 24.11  linux 6.12.3
  43  2025-01-10  NixOS 24.11  linux 6.12.2
  42  2025-01-05  NixOS 24.11  linux 6.12.2
";

    #[test]
    fn test_parse_generations() {
        let gens = GenerationManager::parse_generations(MOCK_GENERATIONS).unwrap();
        assert_eq!(gens.len(), 4);
        assert_eq!(gens[0].number, 45);
        assert!(gens[0].current);
        assert_eq!(gens[1].number, 44);
        assert!(!gens[1].current);
        assert_eq!(gens[3].number, 42);
    }

    #[test]
    fn test_parse_generations_empty() {
        let gens = GenerationManager::parse_generations("").unwrap();
        assert!(gens.is_empty());
    }

    #[test]
    fn test_switch_to_command() {
        let cmd = GenerationManager::switch_to(42);
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nix-env");
        assert!(args.contains(&"42".to_string()));
        assert_eq!(cmd.safety_level(), SafetyLevel::SystemCritical);
    }

    #[test]
    fn test_rollback_command() {
        let cmd = GenerationManager::rollback();
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nixos-rebuild");
        assert!(args.contains(&"--rollback".to_string()));
    }

    #[test]
    fn test_delete_old_command() {
        let cmd = GenerationManager::delete_old(5);
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nix-env");
        assert!(args.contains(&"+5".to_string()));
        assert_eq!(cmd.safety_level(), SafetyLevel::Destructive);
    }

    #[test]
    fn test_delete_older_than_command() {
        let cmd = GenerationManager::delete_older_than(30);
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nix-env");
        assert!(args.contains(&"30d".to_string()));
    }

    #[test]
    fn test_parse_diff() {
        let output = "\
firefox: 120.0 → 121.0, +2.3 MiB
new-package: ∅ → 1.0.0, +5.1 MiB
old-package: 2.0 → ∅, -3.2 MiB
";
        let diff = GenerationManager::parse_diff(output, 42, 43).unwrap();
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added[0], "new-package");
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed[0], "old-package");
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(diff.changed[0].0, "firefox");
    }

    #[test]
    fn test_parse_diff_empty() {
        let diff = GenerationManager::parse_diff("", 1, 2).unwrap();
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());
        assert_eq!(diff.from, 1);
        assert_eq!(diff.to, 2);
    }

    #[test]
    fn test_parse_diff_ascii_arrows() {
        let output = "new-pkg: ∅ -> 1.0, +1 MiB\nold-pkg: 2.0 -> ∅, -1 MiB\n";
        let diff = GenerationManager::parse_diff(output, 10, 11).unwrap();
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.removed.len(), 1);
    }

    #[test]
    fn test_parse_generations_non_numeric_skipped() {
        let output =
            "Header line that is not a generation\n  42  2025-01-05  NixOS 24.11  linux 6.12\n";
        let gens = GenerationManager::parse_generations(output).unwrap();
        assert_eq!(gens.len(), 1);
        assert_eq!(gens[0].number, 42);
    }

    #[test]
    fn test_parse_generations_minimal_tokens() {
        let output = "  99  2025-03-01  (current)\n";
        let gens = GenerationManager::parse_generations(output).unwrap();
        assert_eq!(gens.len(), 1);
        assert!(gens[0].current);
        assert_eq!(gens[0].number, 99);
        assert_eq!(gens[0].date, "2025-03-01");
    }

    #[test]
    fn test_delete_older_than_various() {
        let cmd = GenerationManager::delete_older_than(7);
        let (_, args) = cmd.to_command();
        assert!(args.contains(&"7d".to_string()));

        let cmd2 = GenerationManager::delete_older_than(0);
        let (_, args2) = cmd2.to_command();
        assert!(args2.contains(&"0d".to_string()));
    }

    #[test]
    fn test_switch_to_safety_level() {
        let cmd = GenerationManager::switch_to(1);
        assert_eq!(cmd.safety_level(), SafetyLevel::SystemCritical);
    }

    #[test]
    fn test_rollback_safety_level() {
        let cmd = GenerationManager::rollback();
        assert_eq!(cmd.safety_level(), SafetyLevel::SystemCritical);
    }

    #[test]
    fn test_delete_old_safety_level() {
        let cmd = GenerationManager::delete_old(3);
        assert_eq!(cmd.safety_level(), SafetyLevel::Destructive);
    }
}
