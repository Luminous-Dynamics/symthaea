// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Flake registry observation and tracking.
//!
//! Discovers flake.nix files across the filesystem and parses their metadata
//! via `nix flake metadata --json`. This feeds the world model's understanding
//! of the system's Nix flake ecosystem.

use std::path::Path;
use std::process::Command;
use tracing::debug;

/// Observes registered flakes, their inputs, and upstream update availability.
pub struct FlakeRegistry;

/// Information about a discovered Nix flake.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlakeInfo {
    /// Filesystem path to the flake directory (containing flake.nix).
    pub path: String,
    /// Flake description from metadata, if available.
    pub description: Option<String>,
    /// Names of flake inputs (e.g. ["nixpkgs", "home-manager", "flake-utils"]).
    pub inputs: Vec<String>,
    /// Last modification timestamp as a string, if available.
    pub last_modified: Option<String>,
}

impl FlakeRegistry {
    /// Discover flakes by walking the given search paths looking for `flake.nix` files.
    ///
    /// This performs a shallow walk (up to 3 levels deep) to avoid traversing
    /// the entire filesystem. Skips hidden directories and the nix store.
    pub fn discover(search_paths: &[&str]) -> Result<Vec<FlakeInfo>, std::io::Error> {
        let mut flakes = Vec::new();

        for base_path in search_paths {
            let base = Path::new(base_path);
            if !base.exists() {
                continue;
            }

            Self::walk_for_flakes(base, 0, 3, &mut flakes)?;
        }

        Ok(flakes)
    }

    /// Recursively walk directories up to `max_depth` looking for `flake.nix`.
    fn walk_for_flakes(
        dir: &Path,
        depth: usize,
        max_depth: usize,
        flakes: &mut Vec<FlakeInfo>,
    ) -> Result<(), std::io::Error> {
        if depth > max_depth {
            return Ok(());
        }

        let flake_nix = dir.join("flake.nix");
        if flake_nix.exists() {
            // Found a flake; try to get metadata, fall back to basic info
            let path_str = dir.to_string_lossy().to_string();
            let info = match Self::flake_metadata(&path_str) {
                Ok(info) => info,
                Err(_) => FlakeInfo {
                    path: path_str,
                    description: None,
                    inputs: Vec::new(),
                    last_modified: None,
                },
            };
            flakes.push(info);
            // Don't recurse into flake directories (their sub-dirs are inputs, not separate flakes)
            return Ok(());
        }

        // Read directory entries, skipping hidden dirs and /nix/store
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => return Ok(()), // Permission denied, etc.
        };

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            // Skip hidden directories
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with('.') {
                    continue;
                }
                // Skip nix store and other unproductive paths
                if name == "nix" || name == "proc" || name == "sys" || name == "dev" {
                    continue;
                }
            }

            Self::walk_for_flakes(&path, depth + 1, max_depth, flakes)?;
        }

        Ok(())
    }

    /// Get detailed flake metadata by running `nix flake metadata --json <path>`.
    ///
    /// Parses the JSON output to extract description, inputs, and last-modified timestamp.
    pub fn flake_metadata(path: &str) -> Result<FlakeInfo, std::io::Error> {
        let output = Command::new("nix")
            .args(["flake", "metadata", "--json", path])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "nix flake metadata --json failed for '{}': {}",
                path,
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_flake_metadata_json(path, &stdout)
    }

    /// Parse the JSON output of `nix flake metadata --json`.
    ///
    /// Expected structure:
    /// ```json
    /// {
    ///   "description": "My flake",
    ///   "lastModified": 1705312200,
    ///   "locks": {
    ///     "nodes": {
    ///       "root": {
    ///         "inputs": {
    ///           "nixpkgs": "nixpkgs",
    ///           "home-manager": "home-manager"
    ///         }
    ///       },
    ///       "nixpkgs": { ... },
    ///       "home-manager": { ... }
    ///     }
    ///   },
    ///   "path": "/path/to/flake",
    ///   ...
    /// }
    /// ```
    pub fn parse_flake_metadata_json(
        path: &str,
        json_str: &str,
    ) -> Result<FlakeInfo, std::io::Error> {
        let value: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
            debug!(
                path,
                error = %e,
                json_preview = &json_str[..json_str.len().min(80)],
                "failed to parse flake metadata JSON"
            );
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("failed to parse flake metadata JSON: {e}"),
            )
        })?;

        let description = value
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Extract the resolved path if available, otherwise use the provided path
        let resolved_path = value
            .get("path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| path.to_string());

        // Extract input names from locks.nodes.root.inputs
        let inputs = Self::extract_inputs_from_locks(&value);

        // Extract lastModified timestamp
        let last_modified = value
            .get("lastModified")
            .and_then(|v| v.as_i64())
            .map(|ts| {
                // Convert unix timestamp to human-readable string
                chrono::DateTime::from_timestamp(ts, 0)
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                    .unwrap_or_else(|| ts.to_string())
            });

        Ok(FlakeInfo {
            path: resolved_path,
            description,
            inputs,
            last_modified,
        })
    }

    /// Navigate into `value.locks.nodes.root` — the common prefix for lock queries.
    fn locks_root(value: &serde_json::Value) -> Option<&serde_json::Value> {
        value.get("locks")?.get("nodes")?.get("root")
    }

    /// Extract flake input names from the locks structure.
    fn extract_inputs_from_locks(value: &serde_json::Value) -> Vec<String> {
        let mut inputs = Vec::new();

        if let Some(root) = Self::locks_root(value)
            && let Some(root_inputs) = root.get("inputs")
            && let Some(obj) = root_inputs.as_object()
        {
            for key in obj.keys() {
                inputs.push(key.clone());
            }
        }

        inputs.sort();
        inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOCK_FLAKE_METADATA: &str = r#"{
  "description": "NixOS system configuration",
  "lastModified": 1705312200,
  "locked": {
    "lastModified": 1705312200,
    "narHash": "sha256-abc123",
    "path": "/nix/store/hash-source",
    "type": "path"
  },
  "locks": {
    "nodes": {
      "root": {
        "inputs": {
          "nixpkgs": "nixpkgs",
          "home-manager": "home-manager",
          "flake-utils": "flake-utils"
        }
      },
      "nixpkgs": {
        "locked": {
          "type": "github",
          "owner": "NixOS",
          "repo": "nixpkgs",
          "rev": "abc123"
        }
      },
      "home-manager": {
        "locked": {
          "type": "github",
          "owner": "nix-community",
          "repo": "home-manager",
          "rev": "def456"
        }
      },
      "flake-utils": {
        "locked": {
          "type": "github",
          "owner": "numtide",
          "repo": "flake-utils",
          "rev": "ghi789"
        }
      }
    },
    "root": "root",
    "version": 7
  },
  "originalUrl": "path:/etc/nixos",
  "path": "/etc/nixos",
  "resolved": {
    "type": "path",
    "path": "/etc/nixos"
  },
  "revCount": 42
}"#;

    #[test]
    fn test_parse_flake_metadata_json() {
        let info =
            FlakeRegistry::parse_flake_metadata_json("/etc/nixos", MOCK_FLAKE_METADATA).unwrap();

        assert_eq!(info.path, "/etc/nixos");
        assert_eq!(
            info.description,
            Some("NixOS system configuration".to_string())
        );
        assert_eq!(info.inputs.len(), 3);
        assert!(info.inputs.contains(&"nixpkgs".to_string()));
        assert!(info.inputs.contains(&"home-manager".to_string()));
        assert!(info.inputs.contains(&"flake-utils".to_string()));
        assert!(info.last_modified.is_some());
    }

    #[test]
    fn test_parse_flake_metadata_minimal() {
        let minimal = r#"{"path": "/home/user/project"}"#;
        let info = FlakeRegistry::parse_flake_metadata_json("/home/user/project", minimal).unwrap();

        assert_eq!(info.path, "/home/user/project");
        assert!(info.description.is_none());
        assert!(info.inputs.is_empty());
        assert!(info.last_modified.is_none());
    }

    #[test]
    fn test_parse_flake_metadata_invalid_json() {
        let result = FlakeRegistry::parse_flake_metadata_json("/foo", "not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_flake_metadata_with_description_no_inputs() {
        let json = r#"{
            "description": "A simple flake",
            "path": "/tmp/test-flake",
            "lastModified": 1700000000,
            "locks": {
                "nodes": {
                    "root": {}
                },
                "root": "root",
                "version": 7
            }
        }"#;
        let info = FlakeRegistry::parse_flake_metadata_json("/tmp/test-flake", json).unwrap();
        assert_eq!(info.description, Some("A simple flake".to_string()));
        assert!(info.inputs.is_empty());
        assert!(info.last_modified.is_some());
    }

    #[test]
    fn test_extract_inputs_sorted() {
        // Verify inputs come out sorted regardless of JSON key order
        let json = r#"{
            "path": "/test",
            "locks": {
                "nodes": {
                    "root": {
                        "inputs": {
                            "zzz-last": "zzz",
                            "aaa-first": "aaa",
                            "mmm-middle": "mmm"
                        }
                    }
                }
            }
        }"#;
        let info = FlakeRegistry::parse_flake_metadata_json("/test", json).unwrap();
        assert_eq!(info.inputs, vec!["aaa-first", "mmm-middle", "zzz-last"]);
    }

    #[test]
    fn test_discover_empty_search_paths() {
        let result = FlakeRegistry::discover(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_discover_nonexistent_paths() {
        let result = FlakeRegistry::discover(&["/nonexistent/path/abc123"]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_parse_path_fallback_when_missing() {
        let json = r#"{"description": "no path field"}"#;
        let info = FlakeRegistry::parse_flake_metadata_json("/my/provided/path", json).unwrap();
        assert_eq!(info.path, "/my/provided/path");
    }

    #[test]
    fn test_extract_inputs_no_locks() {
        let value: serde_json::Value = serde_json::from_str(r#"{"path": "/test"}"#).unwrap();
        let inputs = FlakeRegistry::extract_inputs_from_locks(&value);
        assert!(inputs.is_empty());
    }

    #[test]
    fn test_extract_inputs_empty_root_inputs() {
        let value: serde_json::Value =
            serde_json::from_str(r#"{"locks": {"nodes": {"root": {"inputs": {}}}}}"#).unwrap();
        let inputs = FlakeRegistry::extract_inputs_from_locks(&value);
        assert!(inputs.is_empty());
    }

    #[test]
    fn test_parse_flake_metadata_null_values() {
        let json = r#"{
            "description": null,
            "path": "/test",
            "lastModified": null
        }"#;
        let info = FlakeRegistry::parse_flake_metadata_json("/test", json).unwrap();
        assert!(info.description.is_none());
        assert!(info.last_modified.is_none());
    }

    #[test]
    fn test_locks_root_helper() {
        let full: serde_json::Value = serde_json::from_str(MOCK_FLAKE_METADATA).unwrap();
        let root = FlakeRegistry::locks_root(&full);
        assert!(root.is_some());
        assert!(root.unwrap().get("inputs").is_some());
    }

    #[test]
    fn test_locks_root_missing() {
        let no_locks: serde_json::Value = serde_json::from_str(r#"{"path": "/test"}"#).unwrap();
        assert!(FlakeRegistry::locks_root(&no_locks).is_none());

        let no_nodes: serde_json::Value = serde_json::from_str(r#"{"locks": {}}"#).unwrap();
        assert!(FlakeRegistry::locks_root(&no_nodes).is_none());

        let no_root: serde_json::Value =
            serde_json::from_str(r#"{"locks": {"nodes": {}}}"#).unwrap();
        assert!(FlakeRegistry::locks_root(&no_root).is_none());
    }

    #[test]
    fn test_parse_bad_json_returns_error() {
        // Verify that malformed JSON produces an InvalidData error
        let result = FlakeRegistry::parse_flake_metadata_json("/foo", "{truncated");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
