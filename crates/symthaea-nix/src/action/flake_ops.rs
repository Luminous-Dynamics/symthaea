// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Flake Lifecycle Operations
//!
//! Wraps `nix flake` subcommands for flake management:
//! update inputs, lock specific inputs, check, show metadata,
//! and init new flakes. Produces `NixOSCommand` values for
//! state-modifying operations; read-only queries run directly.

use super::executor::{FlakeOperation, NixOSCommand, SafetyLevel};
use std::path::Path;
use std::process::Command;

/// Executes flake-level operations (update inputs, lock, build, check).
pub struct FlakeOps;

/// Metadata about a flake.
#[derive(Debug, Clone)]
pub struct FlakeMetadata {
    /// Resolved URL of the flake.
    pub url: String,
    /// Description from the flake.
    pub description: String,
    /// Input names and their resolved URLs.
    pub inputs: Vec<(String, String)>,
    /// Last modified timestamp.
    pub last_modified: Option<String>,
}

/// Result of a flake check.
#[derive(Debug, Clone)]
pub struct FlakeCheckResult {
    /// Whether the check passed.
    pub passed: bool,
    /// Output messages.
    pub output: String,
    /// Error messages (if any).
    pub errors: String,
}

impl FlakeOps {
    /// Generate a command to update all flake inputs.
    pub fn update_all(flake_dir: &Path) -> NixOSCommand {
        NixOSCommand::Flake {
            operation: FlakeOperation::Update {
                inputs: vec![flake_dir.display().to_string()],
            },
        }
    }

    /// Generate a command to update specific flake inputs.
    pub fn update_inputs(inputs: &[&str]) -> NixOSCommand {
        NixOSCommand::Flake {
            operation: FlakeOperation::Update {
                inputs: inputs.iter().map(|s| s.to_string()).collect(),
            },
        }
    }

    /// Generate a command to lock specific inputs.
    pub fn lock_inputs(inputs: &[&str]) -> NixOSCommand {
        NixOSCommand::Flake {
            operation: FlakeOperation::Lock {
                inputs: inputs.iter().map(|s| s.to_string()).collect(),
            },
        }
    }

    /// Generate a command to show flake outputs.
    pub fn show() -> NixOSCommand {
        NixOSCommand::Flake {
            operation: FlakeOperation::Show,
        }
    }

    /// Generate a command to check a flake.
    pub fn check() -> NixOSCommand {
        NixOSCommand::Flake {
            operation: FlakeOperation::Check,
        }
    }

    /// Get flake metadata (read-only, runs directly).
    pub fn metadata(flake_dir: &Path) -> Result<FlakeMetadata, std::io::Error> {
        let output = Command::new("nix")
            .args(["flake", "metadata", "--json"])
            .current_dir(flake_dir)
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "nix flake metadata failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_metadata_json(&stdout)
    }

    /// List flake inputs from flake.lock (read-only, runs directly).
    pub fn list_inputs(flake_dir: &Path) -> Result<Vec<(String, String)>, std::io::Error> {
        let lock_path = flake_dir.join("flake.lock");
        if !lock_path.exists() {
            return Ok(Vec::new());
        }

        let content = std::fs::read_to_string(&lock_path)?;
        Self::parse_lock_inputs(&content)
    }

    /// Check if a directory contains a flake.
    pub fn is_flake(dir: &Path) -> bool {
        dir.join("flake.nix").exists()
    }

    /// Initialize a new flake in a directory.
    pub fn init(_dir: &Path, template: Option<&str>) -> NixOSCommand {
        let mut args = vec!["flake".to_string(), "init".to_string()];
        if let Some(tmpl) = template {
            args.push("--template".to_string());
            args.push(tmpl.to_string());
        }
        NixOSCommand::Custom {
            command: "nix".to_string(),
            args,
            safety_level: SafetyLevel::UserModify,
        }
    }

    /// Parse flake metadata JSON.
    fn parse_metadata_json(json: &str) -> Result<FlakeMetadata, std::io::Error> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        let url = value["url"].as_str().unwrap_or("").to_string();
        let description = value["description"].as_str().unwrap_or("").to_string();
        let last_modified = value["lastModified"].as_i64().map(|ts| {
            chrono::DateTime::from_timestamp(ts, 0)
                .map(|dt| dt.to_string())
                .unwrap_or_else(|| ts.to_string())
        });

        let mut inputs = Vec::new();
        if let Some(locks) = value["locks"]["nodes"].as_object() {
            for (name, node) in locks {
                if name == "root" {
                    continue;
                }
                let locked_url = node["locked"]["url"]
                    .as_str()
                    .or_else(|| node["locked"]["owner"].as_str())
                    .unwrap_or("")
                    .to_string();
                inputs.push((name.clone(), locked_url));
            }
        }

        Ok(FlakeMetadata {
            url,
            description,
            inputs,
            last_modified,
        })
    }

    /// Parse flake.lock to extract input names and their types.
    fn parse_lock_inputs(lock_content: &str) -> Result<Vec<(String, String)>, std::io::Error> {
        let value: serde_json::Value = serde_json::from_str(lock_content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        let mut inputs = Vec::new();

        if let Some(root_inputs) = value["nodes"]["root"]["inputs"].as_object() {
            for (name, _) in root_inputs {
                let node_type = value["nodes"][name]["locked"]["type"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string();
                inputs.push((name.clone(), node_type));
            }
        }

        Ok(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_all_command() {
        let cmd = FlakeOps::update_all(Path::new("/etc/nixos"));
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nix");
        assert!(args.contains(&"flake".to_string()));
        assert!(args.contains(&"update".to_string()));
    }

    #[test]
    fn test_update_inputs_command() {
        let cmd = FlakeOps::update_inputs(&["nixpkgs", "home-manager"]);
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nix");
        assert!(args.contains(&"update".to_string()));
    }

    #[test]
    fn test_lock_inputs_command() {
        let cmd = FlakeOps::lock_inputs(&["nixpkgs"]);
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nix");
        assert!(args.contains(&"lock".to_string()));
        assert!(args.contains(&"--update-input".to_string()));
    }

    #[test]
    fn test_show_command() {
        let cmd = FlakeOps::show();
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nix");
        assert_eq!(args, vec!["flake", "show"]);
    }

    #[test]
    fn test_check_command() {
        let cmd = FlakeOps::check();
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nix");
        assert_eq!(args, vec!["flake", "check"]);
    }

    #[test]
    fn test_init_command() {
        let cmd = FlakeOps::init(Path::new("."), Some("templates#minimal"));
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nix");
        assert!(args.contains(&"init".to_string()));
        assert!(args.contains(&"--template".to_string()));
    }

    #[test]
    fn test_is_flake() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!FlakeOps::is_flake(dir.path()));

        std::fs::write(dir.path().join("flake.nix"), "{ }").unwrap();
        assert!(FlakeOps::is_flake(dir.path()));
    }

    #[test]
    fn test_parse_lock_inputs() {
        let lock = r#"{
  "nodes": {
    "nixpkgs": {
      "locked": {
        "type": "github",
        "owner": "NixOS",
        "repo": "nixpkgs"
      }
    },
    "home-manager": {
      "locked": {
        "type": "github",
        "owner": "nix-community",
        "repo": "home-manager"
      }
    },
    "root": {
      "inputs": {
        "nixpkgs": "nixpkgs",
        "home-manager": "home-manager"
      }
    }
  },
  "version": 7
}"#;
        let inputs = FlakeOps::parse_lock_inputs(lock).unwrap();
        assert_eq!(inputs.len(), 2);
        let names: Vec<&str> = inputs.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"nixpkgs"));
        assert!(names.contains(&"home-manager"));
    }

    #[test]
    fn test_parse_metadata_json() {
        let json = r#"{
  "url": "path:/etc/nixos",
  "description": "My NixOS configuration",
  "lastModified": 1705000000,
  "locks": {
    "nodes": {
      "root": {},
      "nixpkgs": {
        "locked": {
          "url": "https://github.com/NixOS/nixpkgs"
        }
      }
    }
  }
}"#;
        let meta = FlakeOps::parse_metadata_json(json).unwrap();
        assert_eq!(meta.url, "path:/etc/nixos");
        assert_eq!(meta.description, "My NixOS configuration");
        assert_eq!(meta.inputs.len(), 1);
        assert!(meta.last_modified.is_some());
    }

    #[test]
    fn test_parse_metadata_json_invalid() {
        let result = FlakeOps::parse_metadata_json("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_metadata_json_minimal() {
        let json = r#"{}"#;
        let meta = FlakeOps::parse_metadata_json(json).unwrap();
        assert_eq!(meta.url, "");
        assert_eq!(meta.description, "");
        assert!(meta.inputs.is_empty());
        assert!(meta.last_modified.is_none());
    }

    #[test]
    fn test_parse_metadata_json_owner_fallback() {
        let json = r#"{
  "locks": {
    "nodes": {
      "root": {},
      "nixpkgs": {
        "locked": { "owner": "NixOS", "repo": "nixpkgs" }
      }
    }
  }
}"#;
        let meta = FlakeOps::parse_metadata_json(json).unwrap();
        assert_eq!(meta.inputs.len(), 1);
        assert_eq!(meta.inputs[0].1, "NixOS"); // falls back to owner
    }

    #[test]
    fn test_parse_lock_inputs_invalid_json() {
        let result = FlakeOps::parse_lock_inputs("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_lock_inputs_no_root() {
        let json = r#"{ "nodes": { "nixpkgs": { "locked": { "type": "github" } } } }"#;
        let inputs = FlakeOps::parse_lock_inputs(json).unwrap();
        assert!(inputs.is_empty()); // No root.inputs
    }

    #[test]
    fn test_init_without_template() {
        let cmd = FlakeOps::init(Path::new("."), None);
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nix");
        assert!(args.contains(&"init".to_string()));
        assert!(!args.contains(&"--template".to_string()));
    }

    #[test]
    fn test_update_inputs_empty() {
        let cmd = FlakeOps::update_inputs(&[]);
        let (bin, _args) = cmd.to_command();
        assert_eq!(bin, "nix");
    }

    #[test]
    fn test_list_inputs_nonexistent_dir() {
        let dir = tempfile::tempdir().unwrap();
        // No flake.lock in the dir
        let inputs = FlakeOps::list_inputs(dir.path()).unwrap();
        assert!(inputs.is_empty());
    }
}
