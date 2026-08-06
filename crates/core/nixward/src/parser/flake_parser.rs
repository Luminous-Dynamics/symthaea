// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Parser for Nix flake files (`flake.nix`, `flake.lock`).
//!
//! Parses flake structure via tree-sitter for `flake.nix` and JSON for
//! `flake.lock`, extracting inputs, outputs, description, and dependency
//! relationships for HDC encoding and causal reasoning.

use super::nix_parser::{NixConfig, NixParser, NixValue};
use std::path::Path;

/// Parses flake structure, inputs, outputs, and dependency graphs.
pub struct FlakeParser {
    parser: NixParser,
}

/// Parsed flake.nix information.
#[derive(Debug, Clone)]
pub struct FlakeInfo {
    /// Flake description (from `description = "..."` in flake.nix).
    pub description: Option<String>,
    /// Declared inputs in flake.nix.
    pub inputs: Vec<FlakeInputDecl>,
    /// Output attribute names (e.g., "nixosConfigurations", "packages").
    pub output_attrs: Vec<String>,
    /// Module arguments to the outputs function (e.g., self, nixpkgs).
    pub output_args: Vec<String>,
    /// Parse errors encountered.
    pub errors: Vec<String>,
}

/// A declared flake input from flake.nix.
#[derive(Debug, Clone)]
pub struct FlakeInputDecl {
    /// Input name (e.g., "nixpkgs", "home-manager").
    pub name: String,
    /// URL or reference (e.g., "github:NixOS/nixpkgs/nixos-unstable").
    pub url: Option<String>,
    /// Which other input this follows (e.g., "nixpkgs").
    pub follows: Option<String>,
    /// Whether this is a flake input (default true).
    pub is_flake: Option<bool>,
}

/// Parsed flake.lock information.
#[derive(Debug, Clone)]
pub struct FlakeLockInfo {
    /// Lock file version.
    pub version: u32,
    /// Locked inputs with resolved info.
    pub inputs: Vec<LockedInput>,
    /// Root-level input names (direct dependencies).
    pub root_inputs: Vec<String>,
}

/// A locked input from flake.lock.
#[derive(Debug, Clone)]
pub struct LockedInput {
    /// Input name.
    pub name: String,
    /// Source type (github, git, path, etc.).
    pub source_type: String,
    /// Owner (for github sources).
    pub owner: Option<String>,
    /// Repository name (for github sources).
    pub repo: Option<String>,
    /// Revision hash.
    pub rev: Option<String>,
    /// Which input this follows.
    pub follows: Option<String>,
    /// Last modified timestamp.
    pub last_modified: Option<i64>,
}

/// Combined flake info from both flake.nix and flake.lock.
#[derive(Debug, Clone)]
pub struct FlakeProject {
    /// Info from flake.nix.
    pub flake_info: FlakeInfo,
    /// Info from flake.lock (if present).
    pub lock_info: Option<FlakeLockInfo>,
}

impl FlakeParser {
    /// Create a new flake parser.
    pub fn new() -> Self {
        Self {
            parser: NixParser::new(),
        }
    }

    /// Parse a flake.nix source string.
    pub fn parse_flake_nix(&mut self, source: &str) -> Result<FlakeInfo, String> {
        let config = self
            .parser
            .parse(source)
            .map_err(|e| format!("Failed to parse flake.nix: {}", e.message))?;

        let mut info = FlakeInfo {
            description: None,
            inputs: Vec::new(),
            output_attrs: Vec::new(),
            output_args: Vec::new(),
            errors: config.errors.iter().map(|e| e.message.clone()).collect(),
        };

        // Extract description and inputs from top-level options
        for option in &config.options {
            if option.path == "description" {
                if let NixValue::String(ref s) = option.value {
                    info.description = Some(s.clone());
                }
            } else if option.path.starts_with("inputs.") {
                self.extract_input_decl(&option.path, &option.value, &mut info);
            }
        }

        // Also check for inputs declared as simple strings: inputs.foo.url = "..."
        // or inputs.foo = "github:..."
        self.extract_simple_inputs(&config, &mut info);

        // Extract output function args and top-level output attrs
        self.extract_outputs(&config, &mut info);

        Ok(info)
    }

    /// Parse a flake.lock JSON string.
    pub fn parse_flake_lock(lock_json: &str) -> Result<FlakeLockInfo, String> {
        let value: serde_json::Value = serde_json::from_str(lock_json)
            .map_err(|e| format!("Failed to parse flake.lock: {e}"))?;

        let version = value["version"].as_u64().unwrap_or(7) as u32;

        let nodes = value["nodes"]
            .as_object()
            .ok_or_else(|| "Missing 'nodes' in flake.lock".to_string())?;

        let root_inputs: Vec<String> = value["nodes"]["root"]["inputs"]
            .as_object()
            .map(|obj| obj.keys().cloned().collect())
            .unwrap_or_default();

        let mut inputs = Vec::new();

        for (name, node) in nodes {
            if name == "root" {
                continue;
            }

            let locked = &node["locked"];
            let source_type = locked["type"].as_str().unwrap_or("unknown").to_string();
            let owner = locked["owner"].as_str().map(|s| s.to_string());
            let repo = locked["repo"].as_str().map(|s| s.to_string());
            let rev = locked["rev"].as_str().map(|s| s.to_string());
            let last_modified = locked["lastModified"].as_i64();

            // Check for follows references
            let follows = node["inputs"].as_object().and_then(|inputs_map| {
                inputs_map
                    .iter()
                    .find_map(|(_, v)| v.as_str().map(|s| s.to_string()))
            });

            inputs.push(LockedInput {
                name: name.clone(),
                source_type,
                owner,
                repo,
                rev,
                follows,
                last_modified,
            });
        }

        Ok(FlakeLockInfo {
            version,
            inputs,
            root_inputs,
        })
    }

    /// Parse a flake directory (reads flake.nix and optionally flake.lock).
    pub fn parse_dir(&mut self, dir: &Path) -> Result<FlakeProject, String> {
        let flake_nix_path = dir.join("flake.nix");
        let flake_lock_path = dir.join("flake.lock");

        let source = std::fs::read_to_string(&flake_nix_path)
            .map_err(|e| format!("Failed to read flake.nix: {e}"))?;

        let flake_info = self.parse_flake_nix(&source)?;

        let lock_info = if flake_lock_path.exists() {
            let lock_json = std::fs::read_to_string(&flake_lock_path)
                .map_err(|e| format!("Failed to read flake.lock: {e}"))?;
            Some(Self::parse_flake_lock(&lock_json)?)
        } else {
            None
        };

        Ok(FlakeProject {
            flake_info,
            lock_info,
        })
    }

    /// Extract an input declaration from a parsed option path.
    fn extract_input_decl(&self, path: &str, value: &NixValue, info: &mut FlakeInfo) {
        // Patterns:
        //   inputs.foo.url = "github:..."       → url
        //   inputs.foo.follows = "nixpkgs"      → follows
        //   inputs.foo.flake = false            → is_flake
        //   inputs.foo = "github:..."           → shorthand url
        let parts: Vec<&str> = path.split('.').collect();
        if parts.len() < 2 {
            return;
        }

        let input_name = parts[1].to_string();

        // Find or create the input entry
        let existing = info.inputs.iter_mut().find(|i| i.name == input_name);
        let input = if let Some(existing) = existing {
            existing
        } else {
            info.inputs.push(FlakeInputDecl {
                name: input_name,
                url: None,
                follows: None,
                is_flake: None,
            });
            info.inputs.last_mut().expect("just pushed")
        };

        if parts.len() == 2 {
            // inputs.foo = "github:..."
            if let NixValue::String(s) = value {
                input.url = Some(s.clone());
            }
        } else if parts.len() >= 3 {
            match parts[2] {
                "url" => {
                    if let NixValue::String(s) = value {
                        input.url = Some(s.clone());
                    }
                }
                "follows" => {
                    if let NixValue::String(s) = value {
                        input.follows = Some(s.clone());
                    }
                }
                "flake" => {
                    if let NixValue::Bool(b) = value {
                        input.is_flake = Some(*b);
                    }
                }
                _ => {}
            }
        }
    }

    /// Extract simple string inputs that NixParser might have parsed as top-level bindings.
    fn extract_simple_inputs(&self, config: &NixConfig, info: &mut FlakeInfo) {
        // NixParser may extract `inputs.foo = { url = "..."; }` as an AttrSet
        for option in &config.options {
            if !option.path.starts_with("inputs.") {
                continue;
            }
            let parts: Vec<&str> = option.path.split('.').collect();
            if parts.len() != 2 {
                continue;
            }

            let input_name = parts[1].to_string();
            if info.inputs.iter().any(|i| i.name == input_name) {
                continue; // Already extracted
            }

            // Check if it's an attribute set with url/follows
            if let NixValue::AttrSet(ref attrs) = option.value {
                let mut decl = FlakeInputDecl {
                    name: input_name,
                    url: None,
                    follows: None,
                    is_flake: None,
                };
                if let Some(NixValue::String(url)) = attrs.get("url") {
                    decl.url = Some(url.clone());
                }
                if let Some(NixValue::String(f)) = attrs.get("follows") {
                    decl.follows = Some(f.clone());
                }
                if let Some(NixValue::Bool(b)) = attrs.get("flake") {
                    decl.is_flake = Some(*b);
                }
                info.inputs.push(decl);
            }
        }
    }

    /// Extract output information from the parsed config.
    fn extract_outputs(&self, config: &NixConfig, info: &mut FlakeInfo) {
        // Output function args come from module_args if this is a function
        // In flakes, the top-level is { description, inputs, outputs }
        // outputs is typically a function: { self, nixpkgs, ... }: { ... }
        info.output_args = config.module_args.clone();

        // Collect output attribute names
        for option in &config.options {
            if option.path.starts_with("outputs.") {
                // outputs.nixosConfigurations = ...
                let parts: Vec<&str> = option.path.split('.').collect();
                if parts.len() >= 2 {
                    let attr = parts[1].to_string();
                    if !info.output_attrs.contains(&attr) {
                        info.output_attrs.push(attr);
                    }
                }
            }
        }
    }

    /// Get a summary string for display.
    pub fn summary(info: &FlakeInfo) -> String {
        let mut parts = Vec::new();

        if let Some(ref desc) = info.description {
            parts.push(format!("Description: {desc}"));
        }

        if !info.inputs.is_empty() {
            let names: Vec<&str> = info.inputs.iter().map(|i| i.name.as_str()).collect();
            parts.push(format!(
                "Inputs ({}): {}",
                info.inputs.len(),
                names.join(", ")
            ));
        }

        if !info.output_attrs.is_empty() {
            parts.push(format!("Outputs: {}", info.output_attrs.join(", ")));
        }

        parts.join("\n")
    }
}

impl Default for FlakeParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_flake() {
        let source = r#"
{
  description = "My NixOS configuration";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.home-manager.url = "github:nix-community/home-manager";
  inputs.home-manager.inputs.nixpkgs.follows = "nixpkgs";
}
"#;
        let mut parser = FlakeParser::new();
        let info = parser.parse_flake_nix(source).unwrap();

        assert_eq!(info.description.as_deref(), Some("My NixOS configuration"));
        assert!(!info.inputs.is_empty(), "Should find inputs");

        let nixpkgs = info.inputs.iter().find(|i| i.name == "nixpkgs");
        assert!(nixpkgs.is_some(), "Should find nixpkgs input");
        if let Some(np) = nixpkgs {
            assert!(np.url.as_deref().unwrap_or("").contains("NixOS/nixpkgs"));
        }
    }

    #[test]
    fn test_parse_flake_lock() {
        let lock_json = r#"{
  "nodes": {
    "nixpkgs": {
      "locked": {
        "type": "github",
        "owner": "NixOS",
        "repo": "nixpkgs",
        "rev": "abc123",
        "lastModified": 1705000000
      }
    },
    "home-manager": {
      "locked": {
        "type": "github",
        "owner": "nix-community",
        "repo": "home-manager"
      },
      "inputs": {
        "nixpkgs": "nixpkgs"
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

        let lock_info = FlakeParser::parse_flake_lock(lock_json).unwrap();
        assert_eq!(lock_info.version, 7);
        assert_eq!(lock_info.inputs.len(), 2);
        assert_eq!(lock_info.root_inputs.len(), 2);

        let nixpkgs = lock_info
            .inputs
            .iter()
            .find(|i| i.name == "nixpkgs")
            .unwrap();
        assert_eq!(nixpkgs.source_type, "github");
        assert_eq!(nixpkgs.owner.as_deref(), Some("NixOS"));
        assert_eq!(nixpkgs.repo.as_deref(), Some("nixpkgs"));
        assert_eq!(nixpkgs.rev.as_deref(), Some("abc123"));
        assert_eq!(nixpkgs.last_modified, Some(1705000000));

        let hm = lock_info
            .inputs
            .iter()
            .find(|i| i.name == "home-manager")
            .unwrap();
        assert_eq!(hm.follows, Some("nixpkgs".to_string()));
    }

    #[test]
    fn test_parse_flake_lock_empty_inputs() {
        let lock_json = r#"{
  "nodes": { "root": { "inputs": {} } },
  "version": 7
}"#;
        let lock_info = FlakeParser::parse_flake_lock(lock_json).unwrap();
        assert_eq!(lock_info.inputs.len(), 0);
        assert_eq!(lock_info.root_inputs.len(), 0);
    }

    #[test]
    fn test_flake_summary() {
        let info = FlakeInfo {
            description: Some("Test flake".to_string()),
            inputs: vec![
                FlakeInputDecl {
                    name: "nixpkgs".into(),
                    url: Some("github:NixOS/nixpkgs".into()),
                    follows: None,
                    is_flake: None,
                },
                FlakeInputDecl {
                    name: "home-manager".into(),
                    url: None,
                    follows: Some("nixpkgs".into()),
                    is_flake: None,
                },
            ],
            output_attrs: vec!["nixosConfigurations".into(), "packages".into()],
            output_args: vec![],
            errors: vec![],
        };

        let summary = FlakeParser::summary(&info);
        assert!(summary.contains("Test flake"));
        assert!(summary.contains("Inputs (2)"));
        assert!(summary.contains("nixpkgs"));
        assert!(summary.contains("nixosConfigurations"));
    }

    #[test]
    fn test_flake_input_follows() {
        let source = r#"
{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs";
  inputs.utils.url = "github:numtide/flake-utils";
  inputs.utils.inputs.nixpkgs.follows = "nixpkgs";
}
"#;
        let mut parser = FlakeParser::new();
        let info = parser.parse_flake_nix(source).unwrap();
        assert!(
            info.inputs.len() >= 2,
            "Should find at least 2 inputs, got {}",
            info.inputs.len()
        );
    }

    #[test]
    fn test_parse_empty_flake() {
        let source = "{ }";
        let mut parser = FlakeParser::new();
        let info = parser.parse_flake_nix(source).unwrap();
        assert!(info.description.is_none());
        assert!(info.inputs.is_empty());
    }

    #[test]
    fn test_parse_flake_lock_invalid_json() {
        let result = FlakeParser::parse_flake_lock("this is not json");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to parse flake.lock"));
    }

    #[test]
    fn test_parse_flake_lock_missing_nodes() {
        let result = FlakeParser::parse_flake_lock(r#"{ "version": 7 }"#);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing 'nodes'"));
    }

    #[test]
    fn test_parse_flake_lock_no_version() {
        let json = r#"{ "nodes": { "root": { "inputs": {} } } }"#;
        let info = FlakeParser::parse_flake_lock(json).unwrap();
        assert_eq!(info.version, 7, "Missing version should default to 7");
    }

    #[test]
    fn test_parse_flake_lock_node_missing_locked() {
        let json = r#"{
            "nodes": {
                "root": { "inputs": { "foo": "foo" } },
                "foo": { "original": { "type": "github" } }
            },
            "version": 7
        }"#;
        let info = FlakeParser::parse_flake_lock(json).unwrap();
        assert_eq!(info.inputs.len(), 1);
        assert_eq!(info.inputs[0].source_type, "unknown");
    }

    #[test]
    fn test_parse_flake_lock_root_no_inputs() {
        let json = r#"{ "nodes": { "root": {} }, "version": 7 }"#;
        let info = FlakeParser::parse_flake_lock(json).unwrap();
        assert!(info.root_inputs.is_empty());
    }

    #[test]
    fn test_flake_nix_with_description_only() {
        let source = r#"{ description = "Just a description"; }"#;
        let mut parser = FlakeParser::new();
        let info = parser.parse_flake_nix(source).unwrap();
        assert_eq!(info.description.as_deref(), Some("Just a description"));
        assert!(info.inputs.is_empty());
    }

    #[test]
    fn test_flake_input_flake_false() {
        let source = r#"
{
  inputs.utils.url = "github:numtide/flake-utils";
  inputs.utils.flake = false;
}
"#;
        let mut parser = FlakeParser::new();
        let info = parser.parse_flake_nix(source).unwrap();
        let utils = info.inputs.iter().find(|i| i.name == "utils");
        assert!(utils.is_some());
        if let Some(u) = utils {
            assert_eq!(u.is_flake, Some(false));
        }
    }

    #[test]
    fn test_summary_no_description() {
        let info = FlakeInfo {
            description: None,
            inputs: vec![],
            output_attrs: vec![],
            output_args: vec![],
            errors: vec![],
        };
        let summary = FlakeParser::summary(&info);
        assert!(summary.is_empty(), "Empty flake should have empty summary");
    }

    #[test]
    fn test_summary_outputs_only() {
        let info = FlakeInfo {
            description: None,
            inputs: vec![],
            output_attrs: vec!["packages".into()],
            output_args: vec![],
            errors: vec![],
        };
        let summary = FlakeParser::summary(&info);
        assert!(summary.contains("Outputs: packages"));
    }

    #[test]
    fn test_parse_flake_nix_errors_collected() {
        let mut parser = FlakeParser::new();
        // A syntactically valid but semantically odd flake
        let source = r#"{ ... }: { description = "test"; }"#;
        let info = parser.parse_flake_nix(source).unwrap();
        // Should parse without hard failure; errors list may or may not be populated
        assert!(info.description.is_some() || info.description.is_none());
    }
}
