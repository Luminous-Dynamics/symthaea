// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Configuration Management
//!
//! Provides configuration loading from TOML files and environment variables.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Network configuration
    pub network: NetworkConfig,

    /// Storage configuration
    pub storage: StorageConfig,

    /// PoGQ configuration
    pub pogq: PoGQConfig,

    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// DHT bootstrap nodes
    pub bootstrap_nodes: Vec<String>,

    /// Listen address
    pub listen_addr: String,

    /// Port
    pub port: u16,

    /// Enable mDNS discovery
    pub enable_mdns: bool,
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Storage backend type ("memory", "ipfs", "filecoin")
    pub backend: String,

    /// IPFS API URL (if using IPFS)
    pub ipfs_api_url: Option<String>,

    /// Local storage path
    pub local_path: PathBuf,

    /// Cache size in MB
    pub cache_size_mb: usize,
}

/// PoGQ configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoGQConfig {
    /// Byzantine fault tolerance threshold (0.0 - 1.0)
    pub bft_threshold: f64,

    /// Minimum quality score for validity
    pub min_quality_score: f64,

    /// Number of validation rounds
    pub validation_rounds: usize,

    /// Enable adaptive threshold
    pub adaptive_threshold: bool,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level ("debug", "info", "warn", "error")
    pub level: String,

    /// Output format ("text", "json")
    pub format: String,

    /// Log to file
    pub file: Option<PathBuf>,

    /// Enable performance metrics
    pub metrics: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            network: NetworkConfig::default(),
            storage: StorageConfig::default(),
            pogq: PoGQConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bootstrap_nodes: vec![],
            listen_addr: "0.0.0.0".to_string(),
            port: 9000,
            enable_mdns: true,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: "memory".to_string(),
            ipfs_api_url: None,
            local_path: PathBuf::from(".mycelix/data"),
            cache_size_mb: 100,
        }
    }
}

impl Default for PoGQConfig {
    fn default() -> Self {
        Self {
            bft_threshold: crate::DEFAULT_BFT_THRESHOLD,
            min_quality_score: 0.7,
            validation_rounds: 3,
            adaptive_threshold: true,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "text".to_string(),
            file: None,
            metrics: false,
        }
    }
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::Generic(format!("Failed to read config file: {}", e)))?;

        let config: Config = toml::from_str(&content)
            .map_err(|e| Error::Generic(format!("Failed to parse config: {}", e)))?;

        Ok(config)
    }

    /// Load configuration with environment variable overrides
    pub fn load() -> Result<Self> {
        // Try to load from default locations
        let config_paths = vec![
            PathBuf::from(".mycelix/config.toml"),
            PathBuf::from("config.toml"),
            dirs::config_dir()
                .map(|p| p.join("mycelix-desci/config.toml"))
                .unwrap_or_default(),
        ];

        let mut config = None;
        for path in config_paths {
            if path.exists() {
                config = Some(Self::from_file(&path)?);
                break;
            }
        }

        let mut config = config.unwrap_or_default();

        // Override with environment variables
        config.apply_env_overrides();

        Ok(config)
    }

    /// Apply environment variable overrides
    fn apply_env_overrides(&mut self) {
        if let Ok(level) = std::env::var("MYCELIX_LOG_LEVEL") {
            self.logging.level = level;
        }

        if let Ok(backend) = std::env::var("MYCELIX_STORAGE_BACKEND") {
            self.storage.backend = backend;
        }

        if let Ok(ipfs_url) = std::env::var("MYCELIX_IPFS_API_URL") {
            self.storage.ipfs_api_url = Some(ipfs_url);
        }

        if let Ok(port) = std::env::var("MYCELIX_PORT") {
            if let Ok(port_num) = port.parse() {
                self.network.port = port_num;
            }
        }
    }

    /// Save configuration to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| Error::Generic(format!("Failed to serialize config: {}", e)))?;

        std::fs::write(path, content)
            .map_err(|e| Error::Generic(format!("Failed to write config file: {}", e)))?;

        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate PoGQ threshold
        if self.pogq.bft_threshold < 0.0 || self.pogq.bft_threshold > 1.0 {
            return Err(Error::Generic(
                "PoGQ BFT threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.pogq.min_quality_score < 0.0 || self.pogq.min_quality_score > 1.0 {
            return Err(Error::Generic(
                "PoGQ min quality score must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Validate log level
        let valid_levels = ["debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.logging.level.as_str()) {
            return Err(Error::Generic(format!(
                "Invalid log level: {}. Must be one of: {:?}",
                self.logging.level, valid_levels
            )));
        }

        // Validate storage backend
        let valid_backends = ["memory", "ipfs", "filecoin"];
        if !valid_backends.contains(&self.storage.backend.as_str()) {
            return Err(Error::Generic(format!(
                "Invalid storage backend: {}. Must be one of: {:?}",
                self.storage.backend, valid_backends
            )));
        }

        Ok(())
    }

    /// Initialize default configuration directory and file
    pub fn init_default() -> Result<PathBuf> {
        let config_dir = PathBuf::from(".mycelix");
        let config_file = config_dir.join("config.toml");

        // Create directory if it doesn't exist
        std::fs::create_dir_all(&config_dir)
            .map_err(|e| Error::Generic(format!("Failed to create config directory: {}", e)))?;

        // Create data directory
        let data_dir = config_dir.join("data");
        std::fs::create_dir_all(&data_dir)
            .map_err(|e| Error::Generic(format!("Failed to create data directory: {}", e)))?;

        // Write default config if it doesn't exist
        if !config_file.exists() {
            let default_config = Config::default();
            default_config.save(&config_file)?;
        }

        Ok(config_file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.network.port, 9000);
        assert_eq!(config.storage.backend, "memory");
        assert_eq!(config.pogq.bft_threshold, 0.45);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid BFT threshold
        config.pogq.bft_threshold = 1.5;
        assert!(config.validate().is_err());

        config.pogq.bft_threshold = 0.45;
        assert!(config.validate().is_ok());

        // Invalid log level
        config.logging.level = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml = toml::to_string(&config).unwrap();
        let deserialized: Config = toml::from_str(&toml).unwrap();

        assert_eq!(config.network.port, deserialized.network.port);
        assert_eq!(config.storage.backend, deserialized.storage.backend);
    }
}
