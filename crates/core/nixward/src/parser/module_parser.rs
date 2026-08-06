// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Parser for NixOS module structure and option declarations.
//!
//! Parses NixOS modules using tree-sitter to extract:
//! - Module arguments (config, pkgs, lib, ...)
//! - Option declarations (mkOption, mkEnableOption)
//! - Import lists
//! - Config blocks and their settings
//!
//! This structural information feeds into the causal graph and HDC
//! encoding pipeline for understanding module relationships.

use super::nix_parser::{NixConfig, NixParser};

/// Parses NixOS modules into structured representations of options and config.
pub struct ModuleParser {
    parser: NixParser,
}

/// Parsed NixOS module information.
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    /// Module arguments (e.g., config, pkgs, lib).
    pub module_args: Vec<String>,
    /// Import paths declared in the module.
    pub imports: Vec<String>,
    /// Option declarations (mkOption/mkEnableOption).
    pub option_decls: Vec<OptionDecl>,
    /// Config settings (attribute = value).
    pub config_settings: Vec<ConfigSetting>,
    /// Whether this looks like a NixOS module (has module args pattern).
    pub is_nixos_module: bool,
    /// Parse errors encountered.
    pub errors: Vec<String>,
}

/// A NixOS option declaration extracted from a module.
#[derive(Debug, Clone)]
pub struct OptionDecl {
    /// Full dotted path (e.g., "services.myService.enable").
    pub path: String,
    /// Declared type (e.g., "bool", "str", "listOf str").
    pub option_type: Option<String>,
    /// Default value as string.
    pub default_value: Option<String>,
    /// Description text.
    pub description: Option<String>,
    /// Whether this is an mkEnableOption.
    pub is_enable: bool,
    /// Source line number (1-indexed).
    pub line: usize,
}

/// A config setting from a module's config block.
#[derive(Debug, Clone)]
pub struct ConfigSetting {
    /// Full dotted path (e.g., "services.nginx.enable").
    pub path: String,
    /// Value as string representation.
    pub value: String,
    /// Source line number (1-indexed).
    pub line: usize,
}

impl ModuleParser {
    /// Create a new module parser.
    pub fn new() -> Self {
        Self {
            parser: NixParser::new(),
        }
    }

    /// Parse a NixOS module source string.
    pub fn parse(&mut self, source: &str) -> Result<ModuleInfo, String> {
        let config = self
            .parser
            .parse(source)
            .map_err(|e| format!("Failed to parse module: {}", e.message))?;

        let is_nixos_module = Self::detect_nixos_module(&config);

        let mut info = ModuleInfo {
            module_args: config.module_args.clone(),
            imports: config.imports.clone(),
            option_decls: Vec::new(),
            config_settings: Vec::new(),
            is_nixos_module,
            errors: config.errors.iter().map(|e| e.message.clone()).collect(),
        };

        // Extract option declarations and config settings
        for option in &config.options {
            if Self::is_option_decl(option) {
                info.option_decls.push(Self::parse_option_decl(option));
            } else if option.path.starts_with("config.") || !Self::is_options_path(&option.path) {
                info.config_settings.push(ConfigSetting {
                    path: option.path.clone(),
                    value: option.raw_value.clone(),
                    line: option.line,
                });
            }
        }

        // Also scan raw values for mkOption/mkEnableOption patterns
        Self::scan_for_mk_options(source, &mut info);

        Ok(info)
    }

    /// Parse a NixOS module from a file path.
    pub fn parse_file(&mut self, path: &std::path::Path) -> Result<ModuleInfo, String> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        self.parse(&source)
    }

    /// Detect if this looks like a NixOS module.
    fn detect_nixos_module(config: &NixConfig) -> bool {
        // NixOS modules are functions with { config, pkgs, lib, ... } args
        let has_config = config.module_args.iter().any(|a| a == "config");
        let has_pkgs = config.module_args.iter().any(|a| a == "pkgs");
        let has_lib = config.module_args.iter().any(|a| a == "lib");
        has_lib || (has_config && has_pkgs)
    }

    /// Check if a parsed option looks like an mkOption/mkEnableOption declaration.
    fn is_option_decl(option: &super::nix_parser::NixOption) -> bool {
        let raw = &option.raw_value;
        raw.contains("mkOption")
            || raw.contains("mkEnableOption")
            || option.path.starts_with("options.")
    }

    /// Check if a path looks like it's under the `options` tree.
    fn is_options_path(path: &str) -> bool {
        path.starts_with("options.")
    }

    /// Parse an option declaration from a NixOption.
    fn parse_option_decl(option: &super::nix_parser::NixOption) -> OptionDecl {
        let raw = &option.raw_value;
        let is_enable = raw.contains("mkEnableOption");

        // Try to extract type from mkOption { type = ...; }
        let option_type = Self::extract_field(raw, "type");
        let default_value = Self::extract_field(raw, "default");
        let description = Self::extract_field(raw, "description").or_else(|| {
            // mkEnableOption "description text"
            if is_enable {
                Self::extract_mk_enable_desc(raw)
            } else {
                None
            }
        });

        OptionDecl {
            path: option.path.clone(),
            option_type: if is_enable {
                Some("bool".to_string())
            } else {
                option_type
            },
            default_value: if is_enable {
                Some("false".to_string())
            } else {
                default_value
            },
            description,
            is_enable,
            line: option.line,
        }
    }

    /// Extract a field value from a raw mkOption string.
    fn extract_field(raw: &str, field: &str) -> Option<String> {
        // Simple heuristic: look for `field = value;`
        let pattern = format!("{field} = ");
        if let Some(start) = raw.find(&pattern) {
            let after = &raw[start + pattern.len()..];
            // Find the semicolon that ends this field
            let mut depth = 0;
            let mut end = 0;
            let mut in_string = false;
            for (i, ch) in after.char_indices() {
                match ch {
                    '"' if !in_string => in_string = true,
                    '"' if in_string => in_string = false,
                    '{' | '[' if !in_string => depth += 1,
                    '}' | ']' if !in_string => {
                        if depth > 0 {
                            depth -= 1;
                        } else {
                            end = i;
                            break;
                        }
                    }
                    ';' if !in_string && depth == 0 => {
                        end = i;
                        break;
                    }
                    _ => {}
                }
            }
            if end > 0 {
                let value = after[..end].trim();
                if !value.is_empty() {
                    return Some(value.to_string());
                }
            }
        }
        None
    }

    /// Extract description from mkEnableOption "text".
    fn extract_mk_enable_desc(raw: &str) -> Option<String> {
        if let Some(pos) = raw.find("mkEnableOption") {
            let after = &raw[pos + "mkEnableOption".len()..].trim_start();
            // Look for a quoted string
            if let Some(inner) = after.strip_prefix('"')
                && let Some(end) = inner.find('"')
            {
                return Some(inner[..end].to_string());
            }
        }
        None
    }

    /// Scan source for mkOption/mkEnableOption patterns that the AST might miss.
    fn scan_for_mk_options(source: &str, info: &mut ModuleInfo) {
        for (line_idx, line) in source.lines().enumerate() {
            let trimmed = line.trim();

            // Look for `options.path.to.option = mkOption { ... }` or mkEnableOption
            if (trimmed.contains("mkOption") || trimmed.contains("mkEnableOption"))
                && trimmed.contains('=')
            {
                // Try to extract the path
                if let Some(eq_pos) = trimmed.find('=') {
                    let path_part = trimmed[..eq_pos].trim();
                    // Only add if we don't already have this path
                    if !path_part.is_empty()
                        && !info.option_decls.iter().any(|d| d.path == path_part)
                    {
                        let is_enable = trimmed.contains("mkEnableOption");
                        let desc = if is_enable {
                            Self::extract_mk_enable_desc(trimmed)
                        } else {
                            Self::extract_field(trimmed, "description")
                        };

                        info.option_decls.push(OptionDecl {
                            path: path_part.to_string(),
                            option_type: if is_enable {
                                Some("bool".to_string())
                            } else {
                                Self::extract_field(trimmed, "type")
                            },
                            default_value: if is_enable {
                                Some("false".to_string())
                            } else {
                                Self::extract_field(trimmed, "default")
                            },
                            description: desc,
                            is_enable,
                            line: line_idx + 1,
                        });
                    }
                }
            }
        }
    }

    /// Get a summary string for display.
    pub fn summary(info: &ModuleInfo) -> String {
        let mut parts = Vec::new();

        if info.is_nixos_module {
            parts.push(format!(
                "NixOS module (args: {})",
                info.module_args.join(", ")
            ));
        } else {
            parts.push("Nix expression (not a standard NixOS module)".to_string());
        }

        if !info.imports.is_empty() {
            parts.push(format!(
                "Imports ({}): {}",
                info.imports.len(),
                info.imports.join(", ")
            ));
        }

        if !info.option_decls.is_empty() {
            let paths: Vec<&str> = info.option_decls.iter().map(|d| d.path.as_str()).collect();
            parts.push(format!(
                "Option declarations ({}): {}",
                info.option_decls.len(),
                paths.join(", ")
            ));
        }

        if !info.config_settings.is_empty() {
            parts.push(format!("Config settings: {}", info.config_settings.len()));
        }

        parts.join("\n")
    }
}

impl Default for ModuleParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_module() {
        let source = r#"
{ config, pkgs, lib, ... }: {
  imports = [ ./hardware.nix ];
  services.nginx.enable = true;
  environment.systemPackages = with pkgs; [ vim git ];
}
"#;
        let mut parser = ModuleParser::new();
        let info = parser.parse(source).unwrap();

        assert!(info.is_nixos_module, "Should detect NixOS module pattern");
        assert!(info.module_args.contains(&"config".to_string()));
        assert!(info.module_args.contains(&"pkgs".to_string()));
        assert!(info.module_args.contains(&"lib".to_string()));
        assert!(!info.imports.is_empty(), "Should find imports");
    }

    #[test]
    fn test_detect_non_module() {
        let source = r#"
let
  x = 42;
in
  x + 1
"#;
        let mut parser = ModuleParser::new();
        let info = parser.parse(source).unwrap();
        assert!(
            !info.is_nixos_module,
            "Plain expression should not be detected as module"
        );
    }

    #[test]
    fn test_extract_mk_enable_description() {
        let desc = ModuleParser::extract_mk_enable_desc(r#"mkEnableOption "the nginx web server""#);
        assert_eq!(desc, Some("the nginx web server".to_string()));
    }

    #[test]
    fn test_extract_field_type() {
        let raw = r#"mkOption { type = types.bool; default = false; description = "Enable it"; }"#;
        assert_eq!(
            ModuleParser::extract_field(raw, "type"),
            Some("types.bool".to_string())
        );
        assert_eq!(
            ModuleParser::extract_field(raw, "default"),
            Some("false".to_string())
        );
        assert_eq!(
            ModuleParser::extract_field(raw, "description"),
            Some("\"Enable it\"".to_string())
        );
    }

    #[test]
    fn test_scan_mk_options() {
        let source = r#"
{ config, lib, pkgs, ... }:
let cfg = config.services.myApp;
in {
  options.services.myApp.enable = mkEnableOption "my application";
  options.services.myApp.port = mkOption { type = types.port; default = 8080; };
  config = lib.mkIf cfg.enable {
    systemd.services.myApp = { };
  };
}
"#;
        let mut parser = ModuleParser::new();
        let info = parser.parse(source).unwrap();

        assert!(info.is_nixos_module);
        // Should find mkEnableOption and mkOption via scanning
        let enable_decl = info.option_decls.iter().find(|d| d.path.contains("enable"));
        assert!(
            enable_decl.is_some(),
            "Should find enable option declaration. Found: {:?}",
            info.option_decls
        );
        if let Some(decl) = enable_decl {
            assert!(decl.is_enable, "Should detect mkEnableOption");
        }
    }

    #[test]
    fn test_module_with_imports() {
        let source = r#"
{ config, pkgs, ... }: {
  imports = [
    ./hardware-configuration.nix
    ./networking.nix
  ];
}
"#;
        let mut parser = ModuleParser::new();
        let info = parser.parse(source).unwrap();

        assert!(info.is_nixos_module);
        assert!(
            !info.imports.is_empty(),
            "Should find imports, got: {:?}",
            info.imports
        );
    }

    #[test]
    fn test_summary_output() {
        let info = ModuleInfo {
            module_args: vec!["config".into(), "pkgs".into()],
            imports: vec!["./hardware.nix".into()],
            option_decls: vec![OptionDecl {
                path: "services.foo.enable".into(),
                option_type: Some("bool".into()),
                default_value: Some("false".into()),
                description: Some("Enable foo".into()),
                is_enable: true,
                line: 5,
            }],
            config_settings: vec![],
            is_nixos_module: true,
            errors: vec![],
        };

        let summary = ModuleParser::summary(&info);
        assert!(summary.contains("NixOS module"));
        assert!(summary.contains("config, pkgs"));
        assert!(summary.contains("services.foo.enable"));
    }
}
