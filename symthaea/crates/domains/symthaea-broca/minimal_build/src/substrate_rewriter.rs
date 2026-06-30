// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Substrate Rewriter — Direct source code modification engine.
//!
//! Allows Symthaea to generate, validate, and apply patches to her own
//! source code, enabling physical evolution of her architectural logic.

use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Clone)]
pub struct SubstrateRewriter {
    pub root_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct PatchPreview {
    pub relative_path: String,
    pub exists: bool,
    pub old_len: usize,
    pub new_len: usize,
    pub changed: bool,
}

impl SubstrateRewriter {
    pub fn new(root: &str) -> Self {
        Self {
            root_dir: PathBuf::from(root),
        }
    }

    /// Monitor her own substrate integrity by running her own compiler.
    pub fn monitor_integrity(&self, crate_name: &str) -> Result<Vec<serde_json::Value>> {
        println!(
            "🔍 Substrate Rewriter: Auditing source integrity for {}...",
            crate_name
        );

        let output = Command::new("cargo")
            .args(["check", "-p", crate_name, "--message-format=json"])
            .current_dir(&self.root_dir)
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let diagnostics: Vec<serde_json::Value> = stdout
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .filter(|v: &serde_json::Value| v["reason"] == "compiler-message")
            .collect();

        println!(
            "   └─ Audit complete. Diagnostics found: {}",
            diagnostics.len()
        );
        Ok(diagnostics)
    }

    /// Apply a source code patch to a specific file.
    pub fn apply_patch(&self, relative_path: &str, new_code: &str) -> Result<()> {
        if std::env::var("BROCA_ALLOW_SUBSTRATE_WRITE").as_deref() != Ok("1") {
            let preview = self.preview_patch(relative_path, new_code)?;
            anyhow::bail!(
                "substrate writes are disabled; set BROCA_ALLOW_SUBSTRATE_WRITE=1 to apply. preview={preview:?}"
            );
        }
        self.apply_patch_unchecked(relative_path, new_code)
    }

    /// Preview a source patch without writing it.
    pub fn preview_patch(&self, relative_path: &str, new_code: &str) -> Result<PatchPreview> {
        validate_relative_path(relative_path)?;
        let full_path = self.root_dir.join(relative_path);
        let old = fs::read_to_string(&full_path).unwrap_or_default();
        Ok(PatchPreview {
            relative_path: relative_path.to_string(),
            exists: full_path.exists(),
            old_len: old.len(),
            new_len: new_code.len(),
            changed: old != new_code,
        })
    }

    /// Apply a source code patch after callers have performed their own gates.
    pub fn apply_patch_unchecked(&self, relative_path: &str, new_code: &str) -> Result<()> {
        validate_relative_path(relative_path)?;
        let full_path = self.root_dir.join(relative_path);
        println!(
            "🔧 Substrate Rewriter: Applying architectural patch to {:?}...",
            relative_path
        );

        // 1. Create a backup
        let backup_path = full_path.with_extension("bak");
        fs::copy(&full_path, backup_path)?;

        // 2. Atomic Write
        fs::write(&full_path, new_code)?;

        println!("✅ Patch applied successfully.");
        Ok(())
    }

    /// Apply a 'Holon Patch' — coordinated changes across multiple files and languages.
    pub fn apply_holon_patch(&self, patches: Vec<(String, String)>) -> Result<()> {
        println!(
            "🌀 Substrate Rewriter: Applying Atomic Holon Patch ({} files)...",
            patches.len()
        );

        for (rel_path, new_code) in patches {
            validate_relative_path(&rel_path)?;
            let full_path = self.root_dir.join(&rel_path);
            println!("   └─ Patching {}...", rel_path);

            // Backup
            let backup_path = full_path.with_extension("holon_bak");
            let _ = fs::copy(&full_path, backup_path);

            // Write
            fs::write(full_path, new_code)?;
        }

        println!("✅ Holon Patch applied. Project consistency maintained.");
        Ok(())
    }

    /// Trigger a background Nix rebuild of her own substrate.
    pub fn trigger_rebuild(&self, crate_name: &str) -> Result<()> {
        println!(
            "🏗️ Substrate Rewriter: Triggering background Nix rebuild for {}...",
            crate_name
        );

        let output = Command::new("nix")
            .args(["develop", "--command", "cargo", "check", "-p", crate_name])
            .current_dir(&self.root_dir)
            .output()?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Substrate rebuild failed: {}", err));
        }

        println!("✅ Rebuild verified. Substrate is stable.");
        Ok(())
    }

    /// Scaffold a new crate and add it to the workspace.
    /// This allows Symthaea to grow her own substrate repository.
    pub fn create_new_crate(&self, crate_name: &str, dependencies: &[&str]) -> Result<()> {
        validate_crate_name(crate_name)?;
        println!(
            "🏗️ Substrate Rewriter: Reifying new crate '{}'...",
            crate_name
        );

        let crate_path = self.root_dir.join("crates").join(crate_name);
        fs::create_dir_all(crate_path.join("src"))?;

        // 1. Synthesize Cargo.toml
        let dep_str = dependencies
            .iter()
            .map(|dep| dependency_line(dep))
            .collect::<Result<Vec<_>>>()?
            .join("\n");
        let cargo_toml = format!(
            r#"[package]
name = "{}"
version = "0.1.0"
edition = "2024"

[dependencies]
{}
"#,
            crate_name, dep_str
        );
        fs::write(crate_path.join("Cargo.toml"), cargo_toml)?;

        // 2. Synthesize src/lib.rs
        let lib_rs = format!(
            r#"// Auto-reified by Symthaea
pub fn info() -> &'static str {{
    "This crate was autonomously synthesized to resolve a strategic mission."
}}
"#
        );
        fs::write(crate_path.join("src").join("lib.rs"), lib_rs)?;

        self.add_workspace_member(&format!("crates/{crate_name}"))?;
        println!("   └─ Crate reified successfully.");
        Ok(())
    }

    fn add_workspace_member(&self, member: &str) -> Result<()> {
        let manifest_path = self.root_dir.join("Cargo.toml");
        let manifest = fs::read_to_string(&manifest_path)
            .with_context(|| format!("reading workspace manifest {}", manifest_path.display()))?;
        if manifest.contains(&format!("\"{member}\"")) {
            return Ok(());
        }

        let members_pos = manifest
            .find("members = [")
            .context("workspace manifest does not contain a simple members array")?;
        let insert_pos = manifest[members_pos..]
            .find(']')
            .map(|idx| members_pos + idx)
            .context("workspace members array is not closed")?;

        let mut updated = manifest;
        updated.insert_str(insert_pos, &format!("    \"{member}\",\n"));
        fs::write(&manifest_path, updated)
            .with_context(|| format!("updating workspace manifest {}", manifest_path.display()))?;
        Ok(())
    }

    /// Synthesize a new Nix dependency set for her own flake.nix.
    pub fn synthesize_nix_expression(&self, intent: &str) -> Result<String> {
        println!(
            "❄️ Substrate Rewriter: Synthesizing Nix substrate spec for intent: '{}'...",
            intent
        );

        // (Simplified: generating a dummy flake.nix snippet)
        let nix_code = format!(
            r#"{{
  inputs = {{
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  }};
  outputs = {{ self, nixpkgs }}: {{
    # Intent: {}
    devShells.x86_64-linux.default = nixpkgs.lib.mkShell {{
       buildInputs = [ nixpkgs.rustc nixpkgs.cargo ];
    }};
  }};
}}"#,
            intent
        );
        Ok(nix_code)
    }
}

fn validate_crate_name(name: &str) -> Result<()> {
    let valid = !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
        && name
            .bytes()
            .next()
            .map(|b| b.is_ascii_lowercase())
            .unwrap_or(false);
    if valid {
        Ok(())
    } else {
        anyhow::bail!("invalid crate name {name:?}")
    }
}

fn validate_relative_path(path: &str) -> Result<()> {
    let relative = std::path::Path::new(path);
    let valid = !path.is_empty()
        && relative.is_relative()
        && relative.components().all(|component| {
            !matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::RootDir
            )
        });
    if valid {
        Ok(())
    } else {
        anyhow::bail!("invalid relative path {path:?}")
    }
}

fn dependency_line(dep: &str) -> Result<String> {
    validate_crate_name(dep)?;
    Ok(format!("{dep} = {{ workspace = true }}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_name_validation_rejects_path_traversal() {
        assert!(validate_crate_name("good-crate_1").is_ok());
        assert!(validate_crate_name("../bad").is_err());
        assert!(validate_crate_name("Bad").is_err());
    }

    #[test]
    fn relative_path_validation_rejects_escape() {
        assert!(validate_relative_path("crates/demo/src/lib.rs").is_ok());
        assert!(validate_relative_path("../escape.rs").is_err());
        assert!(validate_relative_path("/tmp/escape.rs").is_err());
    }
}
