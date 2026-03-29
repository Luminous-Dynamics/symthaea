// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! G3: Command Aliases
//!
//! User-defined command shortcuts for common NixOS operations.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;

/// A command alias definition
#[derive(Debug, Clone)]
pub struct Alias {
    /// Alias name (what user types)
    pub name: String,
    /// Expansion (what it becomes)
    pub expansion: String,
    /// Description
    pub description: Option<String>,
    /// Whether this is a built-in alias
    pub builtin: bool,
    /// Argument placeholders ($1, $2, etc.)
    pub accepts_args: bool,
}

impl Alias {
    /// Create a new alias
    pub fn new(name: impl Into<String>, expansion: impl Into<String>) -> Self {
        let expansion_str = expansion.into();
        let accepts_args = expansion_str.contains("$1")
            || expansion_str.contains("$@")
            || expansion_str.contains("${");

        Self {
            name: name.into(),
            expansion: expansion_str,
            description: None,
            builtin: false,
            accepts_args,
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Mark as builtin
    pub fn builtin(mut self) -> Self {
        self.builtin = true;
        self
    }

    /// Expand alias with arguments
    pub fn expand(&self, args: &[&str]) -> String {
        let mut result = self.expansion.clone();

        // Replace numbered args ($1, $2, ...)
        for (i, arg) in args.iter().enumerate() {
            let placeholder = format!("${}", i + 1);
            result = result.replace(&placeholder, arg);
        }

        // Replace $@ with all args
        result = result.replace("$@", &args.join(" "));

        // Replace ${*} with all args
        result = result.replace("${*}", &args.join(" "));

        // Clean up unused placeholders
        for i in (args.len() + 1)..=9 {
            result = result.replace(&format!("${i}"), "");
        }

        result.trim().to_string()
    }
}

/// Alias manager
pub struct AliasManager {
    /// All aliases
    aliases: HashMap<String, Alias>,
    /// Config file path
    config_path: Option<PathBuf>,
    /// Whether aliases have been modified
    modified: bool,
}

impl AliasManager {
    /// Create new alias manager
    pub fn new() -> Self {
        let mut mgr = Self {
            aliases: HashMap::new(),
            config_path: None,
            modified: false,
        };
        mgr.init_builtins();
        mgr
    }

    /// Create with config file
    pub fn with_config(path: impl Into<PathBuf>) -> io::Result<Self> {
        let mut mgr = Self::new();
        mgr.config_path = Some(path.into());
        mgr.load()?;
        Ok(mgr)
    }

    /// Initialize built-in NixOS aliases
    fn init_builtins(&mut self) {
        let builtins = [
            // Package operations
            Alias::new("install", "nix-env -iA nixos.$1")
                .with_description("Install a package")
                .builtin(),
            Alias::new("uninstall", "nix-env -e $1")
                .with_description("Uninstall a package")
                .builtin(),
            Alias::new("search", "nix search nixpkgs#$1")
                .with_description("Search for packages")
                .builtin(),
            Alias::new("upgrade", "sudo nixos-rebuild switch --upgrade")
                .with_description("Upgrade system")
                .builtin(),
            // System operations
            Alias::new("rebuild", "sudo nixos-rebuild switch")
                .with_description("Rebuild NixOS configuration")
                .builtin(),
            Alias::new("test", "sudo nixos-rebuild test")
                .with_description("Test NixOS configuration without switching")
                .builtin(),
            Alias::new("dry", "nixos-rebuild dry-build")
                .with_description("Dry-run build")
                .builtin(),
            Alias::new("boot", "sudo nixos-rebuild boot")
                .with_description("Rebuild for next boot")
                .builtin(),
            Alias::new("rollback", "sudo nixos-rebuild switch --rollback")
                .with_description("Rollback to previous generation")
                .builtin(),
            // Garbage collection
            Alias::new("gc", "sudo nix-collect-garbage")
                .with_description("Collect garbage")
                .builtin(),
            Alias::new("gc-old", "sudo nix-collect-garbage -d")
                .with_description("Delete old generations and collect garbage")
                .builtin(),
            Alias::new("optimize", "sudo nix-store --optimise")
                .with_description("Optimize nix store")
                .builtin(),
            // Shell environments
            Alias::new("shell", "nix-shell -p $@")
                .with_description("Enter shell with packages")
                .builtin(),
            Alias::new("run", "nix run nixpkgs#$1")
                .with_description("Run a package without installing")
                .builtin(),
            Alias::new("develop", "nix develop")
                .with_description("Enter development environment")
                .builtin(),
            // Flakes
            Alias::new("update", "nix flake update")
                .with_description("Update flake inputs")
                .builtin(),
            Alias::new("check", "nix flake check")
                .with_description("Check flake")
                .builtin(),
            Alias::new("show", "nix flake show")
                .with_description("Show flake outputs")
                .builtin(),
            // Information
            Alias::new(
                "generations",
                "sudo nix-env --list-generations -p /nix/var/nix/profiles/system",
            )
            .with_description("List system generations")
            .builtin(),
            Alias::new("version", "nixos-version")
                .with_description("Show NixOS version")
                .builtin(),
            Alias::new("option", "nixos-option $1")
                .with_description("Show option documentation")
                .builtin(),
            // Configuration editing
            Alias::new("edit", "sudo $EDITOR /etc/nixos/configuration.nix")
                .with_description("Edit main configuration")
                .builtin(),
            Alias::new(
                "edit-hw",
                "sudo $EDITOR /etc/nixos/hardware-configuration.nix",
            )
            .with_description("Edit hardware configuration")
            .builtin(),
            Alias::new("edit-flake", "sudo $EDITOR /etc/nixos/flake.nix")
                .with_description("Edit flake.nix")
                .builtin(),
            // Channels (for non-flake systems)
            Alias::new("channels", "sudo nix-channel --list")
                .with_description("List channels")
                .builtin(),
            Alias::new("channel-update", "sudo nix-channel --update")
                .with_description("Update channels")
                .builtin(),
        ];

        for alias in builtins {
            self.aliases.insert(alias.name.clone(), alias);
        }
    }

    /// Add a user alias
    pub fn add(&mut self, alias: Alias) -> bool {
        if alias.builtin {
            return false; // Can't add builtins
        }
        self.aliases.insert(alias.name.clone(), alias);
        self.modified = true;
        true
    }

    /// Remove an alias
    pub fn remove(&mut self, name: &str) -> Option<Alias> {
        if let Some(alias) = self.aliases.get(name) {
            if alias.builtin {
                return None; // Can't remove builtins
            }
        }
        let removed = self.aliases.remove(name);
        if removed.is_some() {
            self.modified = true;
        }
        removed
    }

    /// Get an alias
    pub fn get(&self, name: &str) -> Option<&Alias> {
        self.aliases.get(name)
    }

    /// Check if alias exists
    pub fn exists(&self, name: &str) -> bool {
        self.aliases.contains_key(name)
    }

    /// Expand input if it starts with an alias
    pub fn expand(&self, input: &str) -> Option<String> {
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            return None;
        }

        let name = parts[0];
        let args = &parts[1..];

        self.aliases.get(name).map(|alias| alias.expand(args))
    }

    /// Try to expand, or return original
    pub fn expand_or_original(&self, input: &str) -> String {
        self.expand(input).unwrap_or_else(|| input.to_string())
    }

    /// List all aliases
    pub fn list(&self) -> Vec<&Alias> {
        let mut aliases: Vec<_> = self.aliases.values().collect();
        aliases.sort_by(|a, b| a.name.cmp(&b.name));
        aliases
    }

    /// List built-in aliases
    pub fn list_builtins(&self) -> Vec<&Alias> {
        self.list().into_iter().filter(|a| a.builtin).collect()
    }

    /// List user aliases
    pub fn list_user(&self) -> Vec<&Alias> {
        self.list().into_iter().filter(|a| !a.builtin).collect()
    }

    /// Save user aliases to config file
    pub fn save(&self) -> io::Result<()> {
        if let Some(path) = &self.config_path {
            let user_aliases: Vec<_> = self.list_user();

            let mut content = String::from("# Symthaea Shell Aliases\n");
            content.push_str("# Format: alias=expansion # description\n\n");

            for alias in user_aliases {
                if let Some(desc) = &alias.description {
                    content.push_str(&format!("{}={} # {}\n", alias.name, alias.expansion, desc));
                } else {
                    content.push_str(&format!("{}={}\n", alias.name, alias.expansion));
                }
            }

            fs::write(path, content)?;
        }
        Ok(())
    }

    /// Load aliases from config file
    pub fn load(&mut self) -> io::Result<()> {
        if let Some(path) = &self.config_path {
            if !path.exists() {
                return Ok(());
            }

            let content = fs::read_to_string(path)?;

            for line in content.lines() {
                let line = line.trim();

                // Skip comments and empty lines
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                // Parse alias=expansion # description
                if let Some(eq_pos) = line.find('=') {
                    let name = line[..eq_pos].trim().to_string();
                    let rest = &line[eq_pos + 1..];

                    let (expansion, description) = if let Some(hash_pos) = rest.rfind('#') {
                        (
                            rest[..hash_pos].trim().to_string(),
                            Some(rest[hash_pos + 1..].trim().to_string()),
                        )
                    } else {
                        (rest.trim().to_string(), None)
                    };

                    let mut alias = Alias::new(name, expansion);
                    if let Some(desc) = description {
                        alias = alias.with_description(desc);
                    }
                    self.aliases.insert(alias.name.clone(), alias);
                }
            }
        }
        Ok(())
    }

    /// Check if there are unsaved changes
    pub fn is_modified(&self) -> bool {
        self.modified
    }

    /// Mark as saved
    pub fn mark_saved(&mut self) {
        self.modified = false;
    }
}

impl Default for AliasManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Format alias list for display
pub fn format_alias_list(aliases: &[&Alias], show_builtins: bool) -> String {
    let mut output = String::new();

    let filtered: Vec<_> = aliases
        .iter()
        .filter(|a| show_builtins || !a.builtin)
        .collect();

    if filtered.is_empty() {
        return "No aliases defined.".to_string();
    }

    let max_name_len = filtered.iter().map(|a| a.name.len()).max().unwrap_or(0);

    for alias in filtered {
        let builtin_marker = if alias.builtin { " [builtin]" } else { "" };
        let desc = alias.description.as_deref().unwrap_or("");

        output.push_str(&format!(
            "{:width$} -> {}{}\n",
            alias.name,
            alias.expansion,
            builtin_marker,
            width = max_name_len
        ));

        if !desc.is_empty() {
            output.push_str(&format!(
                "{:width$}    {}\n",
                "",
                desc,
                width = max_name_len
            ));
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alias_expansion() {
        let alias = Alias::new("install", "nix-env -iA nixos.$1");
        assert_eq!(alias.expand(&["firefox"]), "nix-env -iA nixos.firefox");
    }

    #[test]
    fn test_alias_expansion_multiple_args() {
        let alias = Alias::new("shell", "nix-shell -p $@");
        assert_eq!(
            alias.expand(&["git", "vim", "htop"]),
            "nix-shell -p git vim htop"
        );
    }

    #[test]
    fn test_alias_manager() {
        let mgr = AliasManager::new();

        // Builtins should exist
        assert!(mgr.exists("install"));
        assert!(mgr.exists("rebuild"));

        // Should expand
        let expanded = mgr.expand("install firefox").unwrap();
        assert_eq!(expanded, "nix-env -iA nixos.firefox");
    }

    #[test]
    fn test_user_alias() {
        let mut mgr = AliasManager::new();

        let alias = Alias::new("myalias", "echo hello $1");
        mgr.add(alias);

        assert!(mgr.exists("myalias"));
        assert_eq!(mgr.expand("myalias world").unwrap(), "echo hello world");
    }

    #[test]
    fn test_cannot_remove_builtin() {
        let mut mgr = AliasManager::new();
        assert!(mgr.remove("install").is_none());
        assert!(mgr.exists("install"));
    }
}
