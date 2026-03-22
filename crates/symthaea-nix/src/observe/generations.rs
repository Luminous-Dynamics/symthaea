// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS generation observation and history tracking.
//!
//! Parses output from `nix-env --list-generations` and `nix store diff-closures`
//! to provide structured views of system generation history, enabling the world
//! model to understand system evolution over time.

use std::process::Command;

/// Observes system and user profile generations, diffs, and rollback points.
pub struct GenerationObserver;

/// Information about a single NixOS system generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationInfo {
    /// Generation number (monotonically increasing).
    pub number: u64,
    /// Date string when this generation was created (e.g. "2024-01-15 10:30:00").
    pub date: String,
    /// NixOS version string, if parseable from the generation metadata.
    pub nixos_version: Option<String>,
    /// Kernel version, if parseable from the generation metadata.
    pub kernel: Option<String>,
    /// Whether this generation is the currently active one.
    pub current: bool,
}

/// A diff between two generations, showing added and removed store paths.
#[derive(Debug, Clone, Default)]
pub struct GenerationDiff {
    /// Generation number we are diffing from.
    pub from: u64,
    /// Generation number we are diffing to.
    pub to: u64,
    /// Store paths that were added (or upgraded) in the newer generation.
    pub added: Vec<String>,
    /// Store paths that were removed (or downgraded) in the newer generation.
    pub removed: Vec<String>,
    /// Raw diff output for further analysis.
    pub raw: String,
}

impl GenerationObserver {
    /// List all NixOS system generations by running
    /// `nix-env --list-generations -p /nix/var/nix/profiles/system`.
    ///
    /// Each output line has the form:
    /// ```text
    ///    42   2024-01-15 10:30:00   (current)
    /// ```
    pub fn list_generations() -> Result<Vec<GenerationInfo>, std::io::Error> {
        let output = Command::new("nix-env")
            .args(["--list-generations", "-p", "/nix/var/nix/profiles/system"])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "nix-env --list-generations failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_generations(&stdout)
    }

    /// Return the currently booted generation number.
    pub fn current_generation() -> Result<u64, std::io::Error> {
        let generations = Self::list_generations()?;
        generations
            .iter()
            .find(|g| g.current)
            .map(|g| g.number)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "no current generation found in generation list",
                )
            })
    }

    /// Diff two generations by running `nix store diff-closures` on their
    /// profile paths.
    pub fn diff_generations(from: u64, to: u64) -> Result<GenerationDiff, std::io::Error> {
        let from_path = format!("/nix/var/nix/profiles/system-{from}-link");
        let to_path = format!("/nix/var/nix/profiles/system-{to}-link");

        let output = Command::new("nix")
            .args(["store", "diff-closures", &from_path, &to_path])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "nix store diff-closures failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let raw = String::from_utf8_lossy(&output.stdout).to_string();
        Self::parse_diff(from, to, &raw)
    }

    // ---- Parsing helpers (public for testing) ----

    /// Parse the output of `nix-env --list-generations`.
    ///
    /// Expected format per line:
    /// ```text
    ///    42   2024-01-15 10:30:00   (current)
    ///    43   2024-02-01 09:15:22
    /// ```
    pub fn parse_generations(output: &str) -> Result<Vec<GenerationInfo>, std::io::Error> {
        let mut generations = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let current = trimmed.contains("(current)");
            // Remove the "(current)" marker for easier parsing
            let clean = trimmed.replace("(current)", "");
            let parts: Vec<&str> = clean.split_whitespace().collect();

            // We expect at least: number date time
            if parts.len() < 3 {
                continue;
            }

            let number = parts[0].parse::<u64>().map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid generation number '{}': {}", parts[0], e),
                )
            })?;

            let date = format!("{} {}", parts[1], parts[2]);

            generations.push(GenerationInfo {
                number,
                date,
                nixos_version: None,
                kernel: None,
                current,
            });
        }

        Ok(generations)
    }

    /// Parse the output of `nix store diff-closures`.
    ///
    /// Lines that start with '+' are added paths, lines with '-' are removed.
    /// Lines may also look like:
    /// ```text
    /// linux: 6.1.69 → 6.1.72
    /// nginx: 1.24.0 → 1.25.3, +2.1 MiB
    /// ```
    pub fn parse_diff(from: u64, to: u64, raw: &str) -> Result<GenerationDiff, std::io::Error> {
        let mut added = Vec::new();
        let mut removed = Vec::new();

        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Lines with version arrows indicate changes
            if trimmed.contains("\u{2192}") || trimmed.contains("->") {
                // This is an upgrade/change line
                added.push(trimmed.to_string());
            } else if trimmed.starts_with('+') {
                added.push(trimmed.trim_start_matches('+').trim().to_string());
            } else if trimmed.starts_with('-') {
                removed.push(trimmed.trim_start_matches('-').trim().to_string());
            } else {
                // Other format: just record as an addition for now
                added.push(trimmed.to_string());
            }
        }

        Ok(GenerationDiff {
            from,
            to,
            added,
            removed,
            raw: raw.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOCK_GENERATION_LIST: &str = "\
   40   2024-01-10 08:00:00
   41   2024-01-12 14:22:33
   42   2024-01-15 10:30:00   (current)
";

    #[test]
    fn test_parse_generations_basic() {
        let gens = GenerationObserver::parse_generations(MOCK_GENERATION_LIST).unwrap();
        assert_eq!(gens.len(), 3);

        assert_eq!(gens[0].number, 40);
        assert_eq!(gens[0].date, "2024-01-10 08:00:00");
        assert!(!gens[0].current);

        assert_eq!(gens[1].number, 41);
        assert!(!gens[1].current);

        assert_eq!(gens[2].number, 42);
        assert_eq!(gens[2].date, "2024-01-15 10:30:00");
        assert!(gens[2].current);
    }

    #[test]
    fn test_parse_generations_empty() {
        let gens = GenerationObserver::parse_generations("").unwrap();
        assert!(gens.is_empty());
    }

    #[test]
    fn test_parse_generations_single_current() {
        let output = "   1   2024-06-01 12:00:00   (current)\n";
        let gens = GenerationObserver::parse_generations(output).unwrap();
        assert_eq!(gens.len(), 1);
        assert_eq!(gens[0].number, 1);
        assert!(gens[0].current);
    }

    const MOCK_DIFF_OUTPUT: &str = "\
linux: 6.1.69 \u{2192} 6.1.72
nginx: 1.24.0 \u{2192} 1.25.3, +2.1 MiB
-removed-pkg: 1.0.0
+new-pkg: 2.0.0
";

    #[test]
    fn test_parse_diff() {
        let diff = GenerationObserver::parse_diff(41, 42, MOCK_DIFF_OUTPUT).unwrap();
        assert_eq!(diff.from, 41);
        assert_eq!(diff.to, 42);
        // The arrow lines are classified as "added" (changed)
        assert!(diff.added.iter().any(|a| a.contains("linux")));
        assert!(diff.added.iter().any(|a| a.contains("nginx")));
        // Explicit + lines
        assert!(diff.added.iter().any(|a| a.contains("new-pkg")));
        // Explicit - lines
        assert!(diff.removed.iter().any(|r| r.contains("removed-pkg")));
    }

    #[test]
    fn test_parse_diff_empty() {
        let diff = GenerationObserver::parse_diff(1, 2, "").unwrap();
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert_eq!(diff.from, 1);
        assert_eq!(diff.to, 2);
    }
}
