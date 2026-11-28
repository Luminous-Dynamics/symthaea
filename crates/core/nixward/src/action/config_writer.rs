// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS Configuration Writer — Atomic Modifications with Backup
//!
//! Provides safe, atomic configuration file modifications:
//! - Reads existing config via the parser layer
//! - Creates a git-tracked backup before any write
//! - Writes atomically via temp file + rename
//! - Validates syntax with `nix-instantiate --parse` before committing
//!
//! All writes go through the Φ-gated executor — the config writer itself
//! does NOT execute system commands, it only produces modified config text
//! and backup/restore operations.

use crate::parser::nix_parser::NixParser;
use crate::typed_nix::{NixAttributePath, NixExpr, NixPackageRef};

use std::path::{Path, PathBuf};
use std::process::Command;

/// Result of a config write operation.
#[derive(Debug, Clone)]
pub struct WriteResult {
    /// Path that was written.
    pub path: PathBuf,
    /// Backup path (if created).
    pub backup_path: Option<PathBuf>,
    /// Whether the content was actually changed.
    pub changed: bool,
    /// Diff between old and new content (unified format).
    pub diff: String,
}

/// A pending config modification (not yet applied).
#[derive(Debug, Clone)]
pub struct ConfigPatch {
    /// Target file path.
    pub target: PathBuf,
    /// Original content.
    pub original: String,
    /// Modified content.
    pub modified: String,
    /// Human-readable description of the change.
    pub description: String,
}

impl ConfigPatch {
    /// Compute a unified diff between original and modified.
    pub fn diff(&self) -> String {
        let old_lines: Vec<&str> = self.original.lines().collect();
        let new_lines: Vec<&str> = self.modified.lines().collect();

        let mut diff = String::new();
        diff.push_str(&format!("--- {}\n", self.target.display()));
        diff.push_str(&format!("+++ {}\n", self.target.display()));

        // Simple line-by-line diff (not a full unified diff algorithm)
        let max_len = old_lines.len().max(new_lines.len());
        for i in 0..max_len {
            let old = old_lines.get(i).copied().unwrap_or("");
            let new = new_lines.get(i).copied().unwrap_or("");
            if old != new {
                if !old.is_empty() {
                    diff.push_str(&format!("-{old}\n"));
                }
                if !new.is_empty() {
                    diff.push_str(&format!("+{new}\n"));
                }
            } else {
                diff.push_str(&format!(" {old}\n"));
            }
        }

        diff
    }

    /// Whether this patch actually changes anything.
    pub fn is_noop(&self) -> bool {
        self.original == self.modified
    }
}

/// Writes and modifies NixOS configuration files with atomic operations.
pub struct ConfigWriter {
    /// Root directory for NixOS config (default: /etc/nixos).
    config_root: PathBuf,
    /// Whether to create git backups before writes.
    git_backup: bool,
    /// Whether to validate syntax before writing.
    validate: bool,
    /// Dry-run mode: produce patches without writing.
    dry_run: bool,
}

impl ConfigWriter {
    /// Create a new config writer with default settings.
    pub fn new() -> Self {
        Self {
            config_root: PathBuf::from("/etc/nixos"),
            git_backup: true,
            validate: true,
            dry_run: false,
        }
    }

    /// Set the config root directory.
    pub fn with_config_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.config_root = root.into();
        self
    }

    /// Enable or disable git backup.
    pub fn with_git_backup(mut self, enabled: bool) -> Self {
        self.git_backup = enabled;
        self
    }

    /// Enable or disable dry-run mode.
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    fn contains_package_reference(source: &str, package_attr: &str) -> bool {
        let needle = format!("pkgs.{package_attr}");
        let mut search_from = 0usize;
        while let Some(relative) = source[search_from..].find(&needle) {
            let start = search_from + relative;
            let end = start + needle.len();
            let preceding = source[..start].chars().next_back();
            let following = source[end..].chars().next();
            let left_boundary = preceding.is_none_or(|ch| {
                ch.is_whitespace() || matches!(ch, '[' | ',' | '(')
            });
            let right_boundary = following.is_none_or(|ch| {
                ch.is_whitespace() || matches!(ch, ']' | ',' | ';' | ')')
            });
            if left_boundary && right_boundary {
                return true;
            }
            search_from = end;
        }
        false
    }

    /// Add a package to `environment.systemPackages`.
    ///
    /// Returns a patch that, when applied, adds the package to the config.
    pub fn add_system_package(&self, package: &str) -> Result<ConfigPatch, std::io::Error> {
        let package = NixPackageRef::new(package).map_err(|error| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, error.to_string())
        })?;
        let package_attr = package.attribute().as_nix();
        let config_path = self.config_root.join("configuration.nix");
        let original = std::fs::read_to_string(&config_path)?;

        let pkg_entry = format!("    pkgs.{package_attr}");

        // Check if already present
        if Self::contains_package_reference(&original, &package_attr) {
            return Ok(ConfigPatch {
                target: config_path,
                original: original.clone(),
                modified: original,
                description: format!("{package_attr} is already in systemPackages"),
            });
        }

        // Find the systemPackages list and insert a validated package attribute.
        let modified = if let Some(pos) = original.find("environment.systemPackages") {
            if let Some(bracket_pos) = original[pos..].find('[') {
                let insert_at = pos + bracket_pos + 1;
                let mut result = String::with_capacity(original.len() + pkg_entry.len() + 2);
                result.push_str(&original[..insert_at]);
                result.push('\n');
                result.push_str(&pkg_entry);
                result.push_str(&original[insert_at..]);
                result
            } else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Could not find systemPackages list bracket",
                ));
            }
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not find environment.systemPackages in configuration.nix",
            ));
        };

        Ok(ConfigPatch {
            target: config_path,
            original,
            modified,
            description: format!("Add pkgs.{package_attr} to environment.systemPackages"),
        })
    }

    /// Remove a package from `environment.systemPackages`.
    pub fn remove_system_package(&self, package: &str) -> Result<ConfigPatch, std::io::Error> {
        let package = NixPackageRef::new(package).map_err(|error| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, error.to_string())
        })?;
        let package_attr = package.attribute().as_nix();
        let config_path = self.config_root.join("configuration.nix");
        let original = std::fs::read_to_string(&config_path)?;

        let target = format!("pkgs.{package_attr}");
        let mut modified = original.clone();
        let mut search_from = 0usize;
        while let Some(relative) = modified[search_from..].find(&target) {
            let start = search_from + relative;
            let end = start + target.len();
            let preceding = modified[..start].chars().next_back();
            let following = modified[end..].chars().next();

            let left_boundary = preceding.is_none_or(|ch| {
                ch.is_whitespace() || matches!(ch, '[' | ',' | '(')
            });
            let right_boundary = following.is_none_or(|ch| {
                ch.is_whitespace() || matches!(ch, ']' | ',' | ';' | ')')
            });

            if left_boundary && right_boundary {
                modified.replace_range(start..end, "");
                search_from = start;
            } else {
                search_from = end;
            }
        }

        Ok(ConfigPatch {
            target: config_path,
            original,
            modified,
            description: format!("Remove pkgs.{package_attr} from environment.systemPackages"),
        })
    }

    /// Set a NixOS option to a restricted typed literal.
    ///
    /// This compatibility API accepts only booleans, integers, null, and
    /// JSON-quoted strings. Arbitrary Nix source is rejected. Use
    /// `set_option_expr` when a caller already has a typed expression.
    pub fn set_option(
        &self,
        option_path: &str,
        value: &str,
    ) -> Result<ConfigPatch, std::io::Error> {
        let expression = Self::parse_restricted_literal(value)?;
        self.set_option_expr(option_path, &expression)
    }

    /// Parse a deliberately small CLI literal language.
    ///
    /// This does not parse general Nix. It accepts only bool, signed integer,
    /// null and JSON string literals, so a syntactically valid Nix function,
    /// attribute set, interpolation or extra assignment cannot pass through.
    fn parse_restricted_literal(value: &str) -> Result<NixExpr, std::io::Error> {
        let value = value.trim();
        match value {
            "true" => return Ok(NixExpr::Bool(true)),
            "false" => return Ok(NixExpr::Bool(false)),
            "null" => return Ok(NixExpr::Null),
            _ => {}
        }

        if let Ok(integer) = value.parse::<i64>() {
            return Ok(NixExpr::Int(integer));
        }

        if value.starts_with('"') {
            let string = serde_json::from_str::<String>(value).map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("String values must be valid JSON double-quoted strings: {error}"),
                )
            })?;
            return Ok(NixExpr::String(string));
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Unsupported value expression. Use true, false, null, an integer, or a JSON-quoted string; arbitrary Nix source is not accepted.",
        ))
    }

    /// Construct a patch using a validated static option path and typed value.
    ///
    /// The value is rendered from NixExpr, never copied from user source. The
    /// existing file is parsed and exact option paths are matched before a
    /// source line is replaced. This is still a limited source edit: target
    /// option existence/type/writeability and package membership require the
    /// separate target resolver.
    pub fn set_option_expr(
        &self,
        option_path: &str,
        value: &NixExpr,
    ) -> Result<ConfigPatch, std::io::Error> {
        let option_path = NixAttributePath::parse(option_path).map_err(|error| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, error.to_string())
        })?;
        let rendered_path = option_path.as_nix();
        let rendered_value = value.as_nix().map_err(|error| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, error.to_string())
        })?;

        let config_path = self.config_root.join("configuration.nix");
        let original = std::fs::read_to_string(&config_path)?;
        let mut parser = NixParser::new();
        let parsed = parser.parse(&original).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Existing configuration did not parse: {}", error.message),
            )
        })?;
        if !parsed.errors.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Existing configuration contains parse errors: {:?}", parsed.errors),
            ));
        }

        let matching: Vec<_> = parsed
            .options
            .iter()
            .filter(|option| option.path == rendered_path)
            .collect();
        if matching.len() > 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Refusing ambiguous patch: {rendered_path} has multiple definitions"),
            ));
        }

        let option_line = format!("  {rendered_path} = {rendered_value};");
        let modified = if let Some(existing) = matching.first() {
            let same_line_sibling = parsed.options.iter().any(|option| {
                option.line == existing.line && option.path != rendered_path
            });
            if same_line_sibling
                || existing.raw_value.contains('\n')
                || existing.raw_value.contains('\r')
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Refusing line-based edit for {rendered_path}: its source line contains another binding or a multi-line value"
                    ),
                ));
            }

            let mut lines = original.lines().map(str::to_string).collect::<Vec<_>>();
            let line_index = existing.line.saturating_sub(1);
            if line_index >= lines.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Parser returned an invalid source line for {rendered_path}"),
                ));
            }
            let source_line = lines[line_index].trim_start();
            if !source_line.starts_with(&format!("{rendered_path} =")) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Refusing nested/computed edit for {rendered_path}; only direct dotted assignments are supported"
                    ),
                ));
            }

            let indentation = lines[line_index]
                .chars()
                .take_while(|character| character.is_whitespace())
                .collect::<String>();
            lines[line_index] = format!("{indentation}{rendered_path} = {rendered_value};");
            let mut content = lines.join("\n");
            if original.ends_with('\n') {
                content.push('\n');
            }
            content
        } else {
            let segments = rendered_path.split('.').collect::<Vec<_>>();
            let parent_is_defined = (1..segments.len()).any(|end| {
                let parent = segments[..end].join(".");
                parsed.options.iter().any(|option| option.path == parent)
            });
            let child_is_defined = parsed
                .options
                .iter()
                .any(|option| option.path.starts_with(&format!("{rendered_path}.")));
            if parent_is_defined || child_is_defined {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Refusing to add {rendered_path}: existing nested/parent definitions require an AST-aware edit"
                    ),
                ));
            }

            let Some(position) = original.rfind('}') else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Could not locate the final module attribute-set brace",
                ));
            };
            let mut result = String::with_capacity(original.len() + option_line.len() + 2);
            result.push_str(&original[..position]);
            if !result.ends_with('\n') {
                result.push('\n');
            }
            result.push_str(&option_line);
            result.push('\n');
            result.push_str(&original[position..]);
            result
        };

        // Reject any output that the structural parser cannot read.
        let reparsed = parser.parse(&modified).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Generated configuration did not parse: {}", error.message),
            )
        })?;
        if !reparsed.errors.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Generated configuration contains parse errors: {:?}", reparsed.errors),
            ));
        }

        Ok(ConfigPatch {
            target: config_path,
            original,
            modified,
            description: format!("Set {rendered_path} = {rendered_value}"),
        })
    }

    /// Validate that content has balanced braces and doesn't remove critical patterns.
    fn validate_content_structure(original: &str, modified: &str) -> Result<(), std::io::Error> {
        // Check balanced braces in modified content
        let open_braces = modified.chars().filter(|&c| c == '{').count();
        let close_braces = modified.chars().filter(|&c| c == '}').count();
        if open_braces != close_braces {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Unbalanced braces in modified content: {} open, {} close",
                    open_braces, close_braces
                ),
            ));
        }

        let open_brackets = modified.chars().filter(|&c| c == '[').count();
        let close_brackets = modified.chars().filter(|&c| c == ']').count();
        if open_brackets != close_brackets {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Unbalanced brackets in modified content: {} open, {} close",
                    open_brackets, close_brackets
                ),
            ));
        }

        // Detect removal of critical NixOS module imports
        let critical_patterns = ["boot.loader", "fileSystems", "networking.hostName"];

        for pattern in &critical_patterns {
            if original.contains(pattern) && !modified.contains(pattern) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Refusing to remove critical config line containing '{}'",
                        pattern
                    ),
                ));
            }
        }

        Ok(())
    }

    /// Apply a patch: create backup, validate, write atomically.
    pub fn apply_patch(&self, patch: &ConfigPatch) -> Result<WriteResult, std::io::Error> {
        if patch.is_noop() {
            return Ok(WriteResult {
                path: patch.target.clone(),
                backup_path: None,
                changed: false,
                diff: String::new(),
            });
        }

        // Structural validation (always runs, even in dry-run).
        Self::validate_content_structure(&patch.original, &patch.modified)?;

        let expected_target = self.config_root.join("configuration.nix");
        if patch.target != expected_target {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Patch target does not match this ConfigWriter's configuration root",
            ));
        }

        // Bind the preview to the exact on-disk pre-state. A stale diff must
        // never overwrite a concurrent edit.
        let current = std::fs::read_to_string(&patch.target)?;
        if current != patch.original {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Configuration changed after this patch was created; rebuild and review the patch",
            ));
        }

        if self.dry_run {
            // Use the bundled parser for previews so dry-run remains usable on
            // machines without nix-instantiate; it still must be valid Nix syntax.
            let mut parser = NixParser::new();
            let parsed = parser.parse(&patch.modified).map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Dry-run Nix parsing failed: {}", error.message),
                )
            })?;
            if !parsed.errors.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Dry-run Nix parsing reported errors: {:?}", parsed.errors),
                ));
            }

            return Ok(WriteResult {
                path: patch.target.clone(),
                backup_path: None,
                changed: true,
                diff: patch.diff(),
            });
        }

        // Validate syntax with the Nix evaluator/parser before writing.
        if self.validate {
            Self::validate_nix_syntax(&patch.modified)?;
        }

        // Create git backup
        let backup_path = if self.git_backup {
            self.git_commit_backup(&patch.target, &patch.description)?
        } else {
            None
        };

        // Atomic write: write to temp file, then rename
        let temp_path = patch.target.with_extension("nix.tmp");
        std::fs::write(&temp_path, &patch.modified)?;

        // Re-check immediately before rename to narrow the read/modify/write
        // race. This is not a substitute for a cross-process file lock, but it
        // prevents overwriting a target that changed during validation/backup.
        let latest = std::fs::read_to_string(&patch.target)?;
        if latest != patch.original {
            let _ = std::fs::remove_file(&temp_path);
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Configuration changed while the patch was being prepared; no write was performed",
            ));
        }

        if let Err(error) = std::fs::rename(&temp_path, &patch.target) {
            let _ = std::fs::remove_file(&temp_path);
            return Err(error);
        }

        Ok(WriteResult {
            path: patch.target.clone(),
            backup_path,
            changed: true,
            diff: patch.diff(),
        })
    }

    /// Validate Nix syntax using nix-instantiate --parse.
    fn validate_nix_syntax(content: &str) -> Result<(), std::io::Error> {
        let mut child = Command::new("nix-instantiate")
            .args(["--parse", "-"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            stdin.write_all(content.as_bytes())?;
        }

        let output = child.wait_with_output()?;

        if !output.status.success() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Nix syntax validation failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                ),
            ));
        }

        Ok(())
    }

    /// Create a git commit backup of the config directory.
    fn git_commit_backup(
        &self,
        path: &Path,
        message: &str,
    ) -> Result<Option<PathBuf>, std::io::Error> {
        // Check if config root is a git repo
        let git_dir = self.config_root.join(".git");
        if !git_dir.exists() {
            // Initialize git repo
            let status = Command::new("git")
                .args(["init"])
                .current_dir(&self.config_root)
                .status()?;
            if !status.success() {
                return Ok(None);
            }
        }

        // Stage the file
        let rel_path = path.strip_prefix(&self.config_root).unwrap_or(path);

        let _ = Command::new("git")
            .args(["add", &rel_path.display().to_string()])
            .current_dir(&self.config_root)
            .status();

        // Commit
        let commit_msg = format!("nixward backup: {message}");
        let _ = Command::new("git")
            .args(["commit", "-m", &commit_msg, "--allow-empty"])
            .current_dir(&self.config_root)
            .status();

        Ok(Some(self.config_root.join(".git")))
    }

    /// Restore from the last git backup.
    ///
    /// `git_commit_backup` commits the file's BEFORE-state immediately
    /// before each write, so the most recent backup commit (`HEAD`) already
    /// holds the content to restore to — not `HEAD~1`, which skips one
    /// change too far back and, on the very first restore (only one backup
    /// commit exists yet, no parent), fails outright with "unknown revision
    /// HEAD~1". Found via a real integration test against the non-dry-run
    /// write path — see SYMTHAEA_NIXOS_MANAGEMENT_IMPROVEMENT_PLAN_2026-07-26.md
    /// Phase 2.
    pub fn restore_last_backup(&self) -> Result<(), std::io::Error> {
        let status = Command::new("git")
            .args(["checkout", "HEAD", "--", "."])
            .current_dir(&self.config_root)
            .status()?;

        if !status.success() {
            return Err(std::io::Error::other("git restore failed"));
        }

        Ok(())
    }
}

impl Default for ConfigWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn setup_temp_config(content: &str) -> (tempfile::TempDir, ConfigWriter) {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("configuration.nix"), content).unwrap();
        let writer = ConfigWriter::new()
            .with_config_root(dir.path())
            .with_git_backup(false)
            .with_dry_run(true);
        (dir, writer)
    }

    const SAMPLE_CONFIG: &str = r#"{ config, pkgs, ... }:
{
  environment.systemPackages = with pkgs; [
    vim
    git
    firefox
  ];

  services.openssh.enable = true;
  networking.firewall.enable = true;
}
"#;

    #[test]
    fn test_add_package_patch() {
        let (_dir, writer) = setup_temp_config(SAMPLE_CONFIG);
        let patch = writer.add_system_package("htop").unwrap();
        assert!(!patch.is_noop());
        assert!(patch.modified.contains("pkgs.htop"));
        assert!(patch.description.contains("htop"));
    }

    #[test]
    fn add_package_does_not_confuse_longer_attribute_prefix() {
        let config = r#"{ config, pkgs, ... }:
{
  environment.systemPackages = [
    pkgs.github-cli
  ];
}
"#;
        let (_dir, writer) = setup_temp_config(config);
        let patch = writer.add_system_package("git").unwrap();
        assert!(patch.modified.contains("pkgs.git"));
        assert!(patch.modified.contains("pkgs.github-cli"));
    }

    #[test]
    fn test_add_existing_package_noop() {
        let (_dir, writer) = setup_temp_config(SAMPLE_CONFIG);
        let patch = writer.add_system_package("firefox").unwrap();
        // "pkgs.firefox" doesn't appear in the "with pkgs; [" style,
        // but the content does contain "firefox" — the add_system_package
        // checks for "pkgs.firefox" specifically.
        assert!(!patch.modified.contains("pkgs.firefox\n    pkgs.firefox"));
    }

    #[test]
    fn test_remove_package_patch() {
        let config = r#"{ config, pkgs, ... }:
{
  environment.systemPackages = [
    pkgs.vim
    pkgs.git
    pkgs.firefox
  ];
}
"#;
        let (_dir, writer) = setup_temp_config(config);
        let patch = writer.remove_system_package("git").unwrap();
        assert!(!patch.modified.contains("pkgs.git"));
        assert!(patch.modified.contains("pkgs.vim"));
        assert!(patch.modified.contains("pkgs.firefox"));
    }

    #[test]
    fn test_set_option_existing() {
        let (_dir, writer) = setup_temp_config(SAMPLE_CONFIG);
        let patch = writer
            .set_option("services.openssh.enable", "false")
            .unwrap();
        assert!(patch.modified.contains("services.openssh.enable = false;"));
        assert!(!patch.modified.contains("services.openssh.enable = true;"));
    }

    #[test]
    fn test_set_option_new() {
        let (_dir, writer) = setup_temp_config(SAMPLE_CONFIG);
        let patch = writer.set_option("services.nginx.enable", "true").unwrap();
        assert!(patch.modified.contains("services.nginx.enable = true;"));
    }

    #[test]
    fn arbitrary_nix_expressions_are_rejected_by_legacy_set_option() {
        let (_dir, writer) = setup_temp_config(SAMPLE_CONFIG);
        let injection = "true; services.openssh.enable = true";
        let error = writer
            .set_option("networking.firewall.enable", injection)
            .expect_err("arbitrary Nix source must not be accepted");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);

        let error = writer
            .set_option("networking.firewall.enable; services.openssh.enable", "true")
            .expect_err("an injected attribute path must not be accepted");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn json_string_values_are_escaped_as_nix_literals() {
        let (_dir, writer) = setup_temp_config(SAMPLE_CONFIG);
        let payload = concat!("$", "{builtins.abort \"not executable\"}");
        let raw = serde_json::to_string(&payload).unwrap();
        let patch = writer.set_option("networking.hostName", &raw).unwrap();

        assert!(patch.modified.contains(&concat!("\\", "$", "{builtins.abort")));
        let unescaped = patch.modified.replace(&concat!("\\", "$", "{"), "");
        assert!(!unescaped.contains(&concat!("$", "{")));
    }

    #[test]
    fn package_attribute_paths_reject_expression_injection() {
        let (_dir, writer) = setup_temp_config(SAMPLE_CONFIG);
        for candidate in [
            "git; builtins.abort",
            "git ) ; services.openssh.enable = true",
            "python3Packages..numpy",
            "pkgs.git",
            "git with builtins",
        ] {
            let error = writer
                .add_system_package(candidate)
                .expect_err("invalid package attribute must be rejected");
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        }
    }

    #[test]
    fn package_removal_does_not_remove_a_longer_prefix_match() {
        let config = r#"{ config, pkgs, ... }:
{
  environment.systemPackages = [
    pkgs.git
    pkgs.github-cli
  ];
}
"#;
        let (_dir, writer) = setup_temp_config(config);
        let patch = writer.remove_system_package("git").unwrap();
        assert!(!patch.modified.contains("pkgs.git\n"));
        assert!(patch.modified.contains("pkgs.github-cli"));
    }

    #[test]
    fn stale_patch_is_rejected_without_overwriting_concurrent_edit() {
        let (dir, writer) = setup_temp_config(SAMPLE_CONFIG);
        let patch = writer.set_option("services.openssh.enable", "false").unwrap();

        let changed = format!("{SAMPLE_CONFIG}\n# concurrent operator edit\n");
        fs::write(dir.path().join("configuration.nix"), &changed).unwrap();

        let error = writer
            .apply_patch(&patch)
            .expect_err("patch created from a stale pre-state must be rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(
            fs::read_to_string(dir.path().join("configuration.nix")).unwrap(),
            changed
        );
    }

    #[test]
    fn nested_and_composed_option_bindings_are_not_rewritten_as_flat_paths() {
        let nested = r#"{ config, pkgs, ... }:
{
  services = {
    openssh = {
      enable = true;
    };
  };
}
"#;
        let (_dir, writer) = setup_temp_config(nested);
        let error = writer
            .set_option("services.openssh.enable", "false")
            .expect_err("nested binding must require an AST-aware rewrite");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);

        let composed = r#"{ config, pkgs, ... }:
{
  services.openssh.enable = true; services.nginx.enable = true;
}
"#;
        let (_dir, writer) = setup_temp_config(composed);
        let error = writer
            .set_option("services.openssh.enable", "false")
            .expect_err("same-line sibling definitions must not be removed");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn duplicate_option_definitions_are_not_textually_replaced() {
        let duplicate = r#"{ config, pkgs, ... }:
{
  services.openssh.enable = true;
  services.openssh.enable = false;
}
"#;
        let (_dir, writer) = setup_temp_config(duplicate);
        let error = writer
            .set_option("services.openssh.enable", "true")
            .expect_err("ambiguous duplicate definitions must fail closed");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_dry_run_apply() {
        let (dir, writer) = setup_temp_config(SAMPLE_CONFIG);
        let patch = writer.add_system_package("htop").unwrap();
        let result = writer.apply_patch(&patch).unwrap();
        assert!(result.changed);
        assert!(!result.diff.is_empty());

        // Original file should be unchanged in dry-run
        let content = fs::read_to_string(dir.path().join("configuration.nix")).unwrap();
        assert!(!content.contains("pkgs.htop"));
    }

    #[test]
    fn test_validate_content_balanced_braces() {
        let original = "{ config }: { foo = 1; }";
        let modified = "{ config }: { foo = 1;"; // missing closing brace
        let result = ConfigWriter::validate_content_structure(original, modified);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unbalanced braces")
        );
    }

    #[test]
    fn test_validate_content_balanced_brackets() {
        let original = "[ a b c ]";
        let modified = "[ a b c"; // missing closing bracket
        let result = ConfigWriter::validate_content_structure(original, modified);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unbalanced brackets")
        );
    }

    #[test]
    fn test_validate_content_critical_removal_blocked() {
        let original = "{ boot.loader.grub.enable = true; networking.hostName = \"box\"; }";
        let modified = "{ networking.hostName = \"box\"; }"; // removed boot.loader
        let result = ConfigWriter::validate_content_structure(original, modified);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("boot.loader"));
    }

    #[test]
    fn test_validate_content_critical_hostname_removal_blocked() {
        let original = "{ networking.hostName = \"box\"; foo = 1; }";
        let modified = "{ foo = 1; }";
        let result = ConfigWriter::validate_content_structure(original, modified);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("networking.hostName")
        );
    }

    #[test]
    fn test_validate_content_ok() {
        let original = "{ boot.loader.grub.enable = true; }";
        let modified = "{ boot.loader.grub.enable = false; }";
        let result = ConfigWriter::validate_content_structure(original, modified);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_patch_rejects_unbalanced() {
        let (_dir, writer) = setup_temp_config(SAMPLE_CONFIG);
        let patch = ConfigPatch {
            target: PathBuf::from("/tmp/test.nix"),
            original: "{ foo = 1; }".to_string(),
            modified: "{ foo = 1;".to_string(), // unbalanced
            description: "bad patch".to_string(),
        };
        let result = writer.apply_patch(&patch);
        assert!(result.is_err());
    }

    // ---- Real (non-dry-run) write-path integration tests ----
    //
    // Every test above uses `.with_dry_run(true)`, so `apply_patch`'s real
    // branch — nix-instantiate syntax validation, git backup, atomic
    // temp+rename write — had zero test coverage before this module. All
    // tests here run against a `tempfile::tempdir()`, never real
    // `/etc/nixos`. See SYMTHAEA_NIXOS_MANAGEMENT_IMPROVEMENT_PLAN_2026-07-26.md
    // Phase 2.

    // These tests exercise the real nix-instantiate --parse validation path
    // (and, when git_backup=true, real git commands), not a mock. That's
    // the point -- see the module comment above. But nix-instantiate isn't
    // installed in every CI environment that runs `cargo test` (e.g. the
    // plain "Test Feature Combinations" matrix legs, as opposed to the
    // Nix-equipped "Hardened Nix Regressions" job) -- and since nixward is
    // a default workspace member, that made every one of those legs fail
    // regardless of which symthaea feature was actually under test (found
    // 2026-07-27 verifying export-to-standalone.sh against real CI). Skip
    // gracefully rather than fail when the tool genuinely isn't there,
    // rather than fail the whole matrix on an environment gap unrelated to
    // the code under test.
    fn nix_instantiate_available() -> bool {
        Command::new("nix-instantiate")
            .arg("--version")
            .output()
            .is_ok()
    }

    fn setup_temp_config_live(
        content: &str,
        git_backup: bool,
    ) -> Option<(tempfile::TempDir, ConfigWriter)> {
        if !nix_instantiate_available() {
            eprintln!(
                "SKIPPED: nix-instantiate not found on PATH -- this test exercises the real \
                 nix-instantiate --parse validation path and cannot run without it. \
                 Run under `nix develop` or in a Nix-equipped CI job to exercise this test."
            );
            return None;
        }
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("configuration.nix"), content).unwrap();
        let writer = ConfigWriter::new()
            .with_config_root(dir.path())
            .with_git_backup(git_backup)
            .with_dry_run(false);
        Some((dir, writer))
    }

    #[test]
    fn test_apply_patch_writes_atomically_for_real() {
        let Some((dir, writer)) = setup_temp_config_live(SAMPLE_CONFIG, false) else {
            return;
        };
        let patch = writer.add_system_package("htop").unwrap();
        let result = writer.apply_patch(&patch).unwrap();
        assert!(result.changed);

        let on_disk = fs::read_to_string(dir.path().join("configuration.nix")).unwrap();
        assert!(
            on_disk.contains("pkgs.htop"),
            "real (non-dry-run) apply_patch must actually write the new content"
        );
        // The atomic-write temp file must not be left behind.
        assert!(!dir.path().join("configuration.nix.tmp").exists());
    }

    #[test]
    fn test_apply_patch_rejects_invalid_nix_syntax_via_nix_instantiate() {
        // Brace/bracket-balanced but syntactically invalid — the crude
        // structural check in validate_content_structure() cannot catch
        // this; only the real `nix-instantiate --parse` layer can.
        let Some((dir, writer)) = setup_temp_config_live(SAMPLE_CONFIG, false) else {
            return;
        };
        let patch = ConfigPatch {
            target: dir.path().join("configuration.nix"),
            original: SAMPLE_CONFIG.to_string(),
            modified: "{ foo = ; }".to_string(),
            description: "deliberately invalid nix syntax".to_string(),
        };
        let result = writer.apply_patch(&patch);
        assert!(
            result.is_err(),
            "nix-instantiate --parse should reject this even though braces balance"
        );
        assert!(
            result.unwrap_err().to_string().contains("syntax"),
            "error should come from the real Nix syntax validator"
        );

        // And the on-disk file must be untouched — validation runs before
        // the write.
        let on_disk = fs::read_to_string(dir.path().join("configuration.nix")).unwrap();
        assert_eq!(on_disk, SAMPLE_CONFIG);
    }

    #[test]
    fn test_apply_patch_creates_real_git_backup() {
        let Some((dir, writer)) = setup_temp_config_live(SAMPLE_CONFIG, true) else {
            return;
        };
        let patch = writer.add_system_package("htop").unwrap();
        writer.apply_patch(&patch).unwrap();

        assert!(
            dir.path().join(".git").exists(),
            "git repo should be initialized"
        );
        let log = Command::new("git")
            .args(["log", "--oneline"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        let log_str = String::from_utf8_lossy(&log.stdout);
        assert_eq!(
            log_str.lines().count(),
            1,
            "exactly one backup commit should exist after one apply, got: {log_str}"
        );
        assert!(log_str.contains("nixward backup"));
    }

    #[test]
    fn test_restore_last_backup_recovers_immediately_prior_content() {
        // Regression test for a real off-by-one bug found while writing
        // this integration suite: git_commit_backup() commits the
        // BEFORE-state of the file right before each write, so the most
        // recent backup commit (HEAD) already holds the content to restore
        // to. restore_last_backup() used to check out HEAD~1 instead, which
        // skips one change too far back, and on the very first restore
        // (only one commit exists, no parent) failed outright with
        // "unknown revision HEAD~1".
        let Some((dir, writer)) = setup_temp_config_live(SAMPLE_CONFIG, true) else {
            return;
        };

        let patch = writer.add_system_package("htop").unwrap();
        writer.apply_patch(&patch).unwrap();
        let after_first_apply = fs::read_to_string(dir.path().join("configuration.nix")).unwrap();
        assert!(after_first_apply.contains("pkgs.htop"));

        writer
            .restore_last_backup()
            .expect("restore must succeed even with only one backup commit");

        let restored = fs::read_to_string(dir.path().join("configuration.nix")).unwrap();
        assert_eq!(
            restored, SAMPLE_CONFIG,
            "restoring the last backup must recover the content from immediately before the last apply"
        );
    }

    #[test]
    fn test_restore_last_backup_after_two_applies() {
        let Some((dir, writer)) = setup_temp_config_live(SAMPLE_CONFIG, true) else {
            return;
        };

        let patch1 = writer.add_system_package("htop").unwrap();
        writer.apply_patch(&patch1).unwrap();
        let after_first = fs::read_to_string(dir.path().join("configuration.nix")).unwrap();

        let patch2 = writer
            .set_option("services.openssh.enable", "false")
            .unwrap();
        writer.apply_patch(&patch2).unwrap();
        let after_second = fs::read_to_string(dir.path().join("configuration.nix")).unwrap();
        assert_ne!(after_first, after_second);

        writer.restore_last_backup().unwrap();
        let restored = fs::read_to_string(dir.path().join("configuration.nix")).unwrap();
        assert_eq!(
            restored, after_first,
            "restoring after the second apply must recover the state from right after the first apply, not the original"
        );
    }

    #[test]
    fn test_patch_diff() {
        let patch = ConfigPatch {
            target: PathBuf::from("/etc/nixos/configuration.nix"),
            original: "line1\nline2\n".to_string(),
            modified: "line1\nline2_changed\n".to_string(),
            description: "test".to_string(),
        };
        let diff = patch.diff();
        assert!(diff.contains("-line2"));
        assert!(diff.contains("+line2_changed"));
    }
}
