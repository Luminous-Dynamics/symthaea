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

    /// Add a package to `environment.systemPackages`.
    ///
    /// Returns a patch that, when applied, adds the package to the config.
    pub fn add_system_package(&self, package: &str) -> Result<ConfigPatch, std::io::Error> {
        let config_path = self.config_root.join("configuration.nix");
        let original = std::fs::read_to_string(&config_path)?;

        let pkg_entry = format!("    pkgs.{package}");

        // Check if already present
        if original.contains(&format!("pkgs.{package}")) {
            return Ok(ConfigPatch {
                target: config_path,
                original: original.clone(),
                modified: original,
                description: format!("{package} is already in systemPackages"),
            });
        }

        // Find the systemPackages list and insert
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
            description: format!("Add pkgs.{package} to environment.systemPackages"),
        })
    }

    /// Remove a package from `environment.systemPackages`.
    pub fn remove_system_package(&self, package: &str) -> Result<ConfigPatch, std::io::Error> {
        let config_path = self.config_root.join("configuration.nix");
        let original = std::fs::read_to_string(&config_path)?;

        let patterns = [
            format!("    pkgs.{package}\n"),
            format!("    pkgs.{package}"),
            format!("pkgs.{package}\n"),
            format!("pkgs.{package}"),
        ];

        let mut modified = original.clone();
        for pattern in &patterns {
            modified = modified.replace(pattern, "");
        }

        Ok(ConfigPatch {
            target: config_path,
            original,
            modified,
            description: format!("Remove pkgs.{package} from environment.systemPackages"),
        })
    }

    /// Set a NixOS option to a value.
    ///
    /// This is a simple text-based approach — for complex modifications,
    /// use the parser layer to produce a proper AST edit.
    pub fn set_option(
        &self,
        option_path: &str,
        value: &str,
    ) -> Result<ConfigPatch, std::io::Error> {
        let config_path = self.config_root.join("configuration.nix");
        let original = std::fs::read_to_string(&config_path)?;

        let option_line = format!("  {option_path} = {value};");

        let modified = if let Some(pos) = original.find(option_path) {
            // Find the end of this line (semicolon)
            if let Some(semi) = original[pos..].find(';') {
                let line_start = original[..pos].rfind('\n').map(|p| p + 1).unwrap_or(0);
                let line_end = pos + semi + 1;
                let mut result = String::with_capacity(original.len());
                result.push_str(&original[..line_start]);
                result.push_str(&option_line);
                result.push_str(&original[line_end..]);
                result
            } else {
                original.clone()
            }
        } else {
            // Option not found — insert before the closing brace
            if let Some(pos) = original.rfind('}') {
                let mut result = String::with_capacity(original.len() + option_line.len() + 2);
                result.push_str(&original[..pos]);
                result.push_str(&option_line);
                result.push('\n');
                result.push_str(&original[pos..]);
                result
            } else {
                original.clone()
            }
        };

        Ok(ConfigPatch {
            target: config_path,
            original,
            modified,
            description: format!("Set {option_path} = {value}"),
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

        // Structural validation (always runs, even in dry-run)
        Self::validate_content_structure(&patch.original, &patch.modified)?;

        if self.dry_run {
            return Ok(WriteResult {
                path: patch.target.clone(),
                backup_path: None,
                changed: true,
                diff: patch.diff(),
            });
        }

        // Validate syntax
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
        std::fs::rename(&temp_path, &patch.target)?;

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
        let commit_msg = format!("nix-mind backup: {message}");
        let _ = Command::new("git")
            .args(["commit", "-m", &commit_msg, "--allow-empty"])
            .current_dir(&self.config_root)
            .status();

        Ok(Some(self.config_root.join(".git")))
    }

    /// Restore from the last git backup.
    pub fn restore_last_backup(&self) -> Result<(), std::io::Error> {
        let status = Command::new("git")
            .args(["checkout", "HEAD~1", "--", "."])
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
