// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Morphological Bridge — Autonomous evolution of her physical form.
//!
//! Allows Symthaea to refactor her own Soma body models (URDF/XML)
//! to optimize for kinetic performance and topological stability.

use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;

#[derive(Clone)]
pub struct MorphologicalBridge {
    pub soma_root: PathBuf,
}

impl MorphologicalBridge {
    pub fn new(root: &str) -> Self {
        Self {
            soma_root: PathBuf::from(root).join("crates/symthaea-soma"),
        }
    }

    /// Refactor a physical body model (MuJoCo XML).
    pub fn refactor_body_model(&self, model_name: &str, new_xml: &str) -> Result<()> {
        validate_model_name(model_name)?;
        println!(
            "🦾 Morphological Bridge: Refactoring physical body model '{}'...",
            model_name
        );

        let model_path = self
            .soma_root
            .join("models")
            .join(format!("{}.xml", model_name));
        fs::create_dir_all(model_path.parent().context("model path has no parent")?)?;

        // Backup
        if model_path.exists() {
            let backup = model_path.with_extension("morph_bak");
            let _ = fs::copy(&model_path, backup);
        }

        // Write new physical DNA
        fs::write(&model_path, new_xml)?;

        println!("   ✅ Physical DNA update applied to Soma substrate.");
        Ok(())
    }
}

fn validate_model_name(name: &str) -> Result<()> {
    let valid = !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_');
    if valid {
        Ok(())
    } else {
        anyhow::bail!("invalid model name {name:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refactor_rejects_path_traversal() {
        let bridge = MorphologicalBridge::new(".");
        assert!(
            bridge
                .refactor_body_model("../escape", "<mujoco />")
                .is_err()
        );
    }
}
