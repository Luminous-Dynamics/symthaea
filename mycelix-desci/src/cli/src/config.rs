// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Configuration management

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Config {
    /// API base URL
    pub api_url: Option<String>,

    /// Default output format
    pub output_format: Option<String>,

    /// Verbose mode
    pub verbose: Option<bool>,
}

impl Config {
    /// Load configuration from file or create default
    pub fn load_or_default() -> Result<Self> {
        let config_path = Self::default_config_path()?;

        if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)
                .context("Failed to read config file")?;
            toml::from_str(&contents).context("Failed to parse config file")
        } else {
            // Create default config directory
            if let Some(parent) = config_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }

            let config = Config::default();
            config.save()?;
            Ok(config)
        }
    }

    /// Save configuration to file
    pub fn save(&self) -> Result<()> {
        let config_path = Self::default_config_path()?;

        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create config directory")?;
        }

        let contents = toml::to_string_pretty(self)
            .context("Failed to serialize config")?;

        std::fs::write(&config_path, contents)
            .context("Failed to write config file")?;

        Ok(())
    }

    /// Get default config path
    fn default_config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .context("Failed to get config directory")?;

        Ok(config_dir.join("mycelix").join("config.toml"))
    }

    /// Get config path for display
    pub fn config_path(&self) -> String {
        Self::default_config_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    }
}
