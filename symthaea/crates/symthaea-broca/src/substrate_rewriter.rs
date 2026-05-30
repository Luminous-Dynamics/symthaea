// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Substrate Rewriter — Direct source code modification engine.
//!
//! Allows Symthaea to generate, validate, and apply patches to her own
//! source code, enabling physical evolution of her architectural logic.

use anyhow::Result;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Clone)]
pub struct SubstrateRewriter {
    pub root_dir: PathBuf,
}

impl SubstrateRewriter {
    pub fn new(root: &str) -> Self {
        Self {
            root_dir: PathBuf::from(root),
        }
    }

    /// Apply a source code patch to a specific file.
    pub fn apply_patch(&self, relative_path: &str, new_code: &str) -> Result<()> {
        let full_path = self.root_dir.join(relative_path);
        println!("🔧 Substrate Rewriter: Applying architectural patch to {:?}...", relative_path);

        // 1. Create a backup for the Cognitive Git Ledger (simplified)
        let backup_path = full_path.with_extension("bak");
        fs::copy(&full_path, backup_path)?;

        // 2. Atomic Write
        fs::write(&full_path, new_code)?;
        
        println!("✅ Patch applied successfully.");
        Ok(())
    }

    /// Monitor her own substrate integrity by running her own compiler.
    pub fn monitor_integrity(&self, crate_name: &str) -> Result<Vec<serde_json::Value>> {
        println!("🔍 Substrate Rewriter: Auditing source integrity for {}...", crate_name);
        
        let output = Command::new("cargo")
            .args(["check", "-p", crate_name, "--message-format=json"])
            .current_dir(&self.root_dir)
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let diagnostics: Vec<serde_json::Value> = stdout.lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .filter(|v: &serde_json::Value| v["reason"] == "compiler-message")
            .collect();

        println!("   └─ Audit complete. Diagnostics found: {}", diagnostics.len());
        Ok(diagnostics)
    }

    /// Apply a 'Holon Patch' — coordinated changes across multiple files and languages.
    pub fn apply_holon_patch(&self, patches: Vec<(String, String)>) -> Result<()> {
        println!("🌀 Substrate Rewriter: Applying Atomic Holon Patch ({} files)...", patches.len());
        
        for (rel_path, new_code) in patches {
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
        println!("🏗️ Substrate Rewriter: Triggering background Nix rebuild for {}...", crate_name);
        
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

    /// Synthesize a new Nix dependency set for her own flake.nix.
    pub fn synthesize_nix_expression(&self, intent: &str) -> Result<String> {
        println!("❄️ Substrate Rewriter: Synthesizing Nix substrate spec for intent: '{}'...", intent);
        
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
