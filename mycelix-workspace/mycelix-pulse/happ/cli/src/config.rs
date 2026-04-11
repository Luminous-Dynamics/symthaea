// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Configuration for Mycelix Mail CLI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// User identity configuration
    pub identity: IdentityConfig,

    /// Holochain conductor configuration
    pub conductor: ConductorConfig,

    /// External services configuration
    pub services: ServicesConfig,

    /// User preferences
    pub preferences: PreferencesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityConfig {
    /// User's DID
    pub did: Option<String>,

    /// User's email address (optional, for convenience)
    pub email: Option<String>,

    /// Path to private key file
    pub private_key_path: PathBuf,

    /// Path to public key file
    pub public_key_path: PathBuf,

    /// Holochain agent public key
    pub agent_pub_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConductorConfig {
    /// Holochain conductor WebSocket URL
    pub url: String,

    /// Application ID (defaults to "mycelix-mail")
    #[serde(default = "default_app_id")]
    pub app_id: Option<String>,

    /// Connection timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicesConfig {
    /// DID registry URL
    pub did_registry_url: String,

    /// MATL bridge URL
    pub matl_bridge_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferencesConfig {
    /// Default epistemic tier for messages (0-4)
    #[serde(default = "default_tier")]
    pub default_tier: u8,

    /// Auto-sync on startup
    #[serde(default = "default_true")]
    pub auto_sync: bool,

    /// Cache TTL in seconds
    #[serde(default = "default_cache_ttl")]
    pub cache_ttl: u64,

    /// Message display format
    #[serde(default = "default_format")]
    pub display_format: String,
}

// Default value functions
fn default_timeout() -> u64 {
    30
}
fn default_app_id() -> Option<String> {
    Some("mycelix-mail".to_string())
}
fn default_tier() -> u8 {
    2
}
fn default_true() -> bool {
    true
}
fn default_cache_ttl() -> u64 {
    3600
}
fn default_format() -> String {
    "table".to_string()
}

impl Config {
    /// Get the default config directory path
    pub fn config_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Could not determine home directory")?;
        Ok(home.join(".mycelix-mail"))
    }

    /// Get the default config file path
    pub fn config_file() -> Result<PathBuf> {
        Ok(Self::config_dir()?.join("config.toml"))
    }

    /// Get the default keys directory path
    pub fn keys_dir() -> Result<PathBuf> {
        Ok(Self::config_dir()?.join("keys"))
    }

    /// Load configuration from file, or create default if it doesn't exist
    pub fn load_or_create() -> Result<Self> {
        let config_file = Self::config_file()?;

        if config_file.exists() {
            Self::load()
        } else {
            let config = Self::default();
            config.save()?;
            Ok(config)
        }
    }

    /// Load configuration from file
    pub fn load() -> Result<Self> {
        let config_file = Self::config_file()?;
        let contents = fs::read_to_string(&config_file)
            .with_context(|| format!("Failed to read config file: {:?}", config_file))?;

        let config: Config = toml::from_str(&contents).context("Failed to parse config file")?;

        Ok(config)
    }

    /// Save configuration to file
    pub fn save(&self) -> Result<()> {
        let config_dir = Self::config_dir()?;
        let config_file = Self::config_file()?;
        let keys_dir = Self::keys_dir()?;

        // Create directories if they don't exist
        fs::create_dir_all(&config_dir).context("Failed to create config directory")?;
        fs::create_dir_all(&keys_dir).context("Failed to create keys directory")?;

        // Serialize config to TOML
        let contents = toml::to_string_pretty(self).context("Failed to serialize config")?;

        // Write to file
        fs::write(&config_file, contents)
            .with_context(|| format!("Failed to write config file: {:?}", config_file))?;

        Ok(())
    }

    /// Update DID in configuration
    pub fn set_did(&mut self, did: String) -> Result<()> {
        self.identity.did = Some(did);
        self.save()
    }

    /// Update email in configuration
    pub fn set_email(&mut self, email: String) -> Result<()> {
        self.identity.email = Some(email);
        self.save()
    }

    /// Update agent public key
    pub fn set_agent_key(&mut self, agent_key: String) -> Result<()> {
        self.identity.agent_pub_key = Some(agent_key);
        self.save()
    }
}

impl Default for Config {
    fn default() -> Self {
        let keys_dir = Self::keys_dir().unwrap_or_else(|_| PathBuf::from("./keys"));

        Config {
            identity: IdentityConfig {
                did: None,
                email: None,
                private_key_path: keys_dir.join("private.key"),
                public_key_path: keys_dir.join("public.key"),
                agent_pub_key: None,
            },
            conductor: ConductorConfig {
                url: "ws://localhost:8888".to_string(),
                app_id: Some("mycelix_mail".to_string()),
                timeout: default_timeout(),
            },
            services: ServicesConfig {
                did_registry_url: "http://localhost:5000".to_string(),
                matl_bridge_url: "http://localhost:5001".to_string(),
            },
            preferences: PreferencesConfig {
                default_tier: default_tier(),
                auto_sync: default_true(),
                cache_ttl: default_cache_ttl(),
                display_format: default_format(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.conductor.app_id, "mycelix_mail");
        assert_eq!(config.preferences.default_tier, 2);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: Config = toml::from_str(&toml_str).unwrap();

        assert_eq!(config.conductor.url, deserialized.conductor.url);
    }
}
