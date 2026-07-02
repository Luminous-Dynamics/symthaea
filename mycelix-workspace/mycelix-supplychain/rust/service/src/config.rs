// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Configuration management for 12-factor app compliance
//!
//! All configuration is loaded from environment variables with sensible defaults.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::num::NonZeroU32;

/// Complete application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Server configuration
    pub server: ServerConfig,

    /// Database configuration
    pub database: DatabaseConfig,

    /// Security configuration
    pub security: SecurityConfig,

    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,

    /// Observability configuration
    pub observability: ObservabilityConfig,
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Address to bind to (default: 127.0.0.1:8080)
    pub bind_address: SocketAddr,

    /// Maximum request body size in bytes (default: 2MB)
    pub max_body_size: usize,
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Database URL (default: sqlite://data/claims.db)
    pub url: String,

    /// Connection pool max size (default: 10)
    pub max_connections: u32,

    /// Connection timeout in seconds (default: 30)
    pub connect_timeout_seconds: u64,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Allowed CORS origins (comma-separated)
    pub allowed_origins: Vec<String>,

    /// Enable permissive CORS (for development only)
    pub permissive_cors: bool,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per second
    pub requests_per_second: NonZeroU32,

    /// Burst size
    pub burst_size: NonZeroU32,
}

/// Observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Log level (default: info)
    pub log_level: String,

    /// Enable JSON logging (default: false)
    pub json_logs: bool,

    /// Enable request tracing (default: true)
    pub request_tracing: bool,
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let config = Self {
            server: ServerConfig::from_env()?,
            database: DatabaseConfig::from_env()?,
            security: SecurityConfig::from_env()?,
            rate_limit: RateLimitConfig::from_env()?,
            observability: ObservabilityConfig::from_env()?,
        };

        config.validate()?;
        Ok(config)
    }

    /// Validate configuration
    fn validate(&self) -> Result<()> {
        // Validate max body size is reasonable
        if self.server.max_body_size > 100 * 1024 * 1024 {
            anyhow::bail!("MAX_BODY_SIZE too large (max 100MB)");
        }

        // Validate database URL format
        if !self.database.url.starts_with("sqlite://")
            && !self.database.url.starts_with("postgres://")
            && !self.database.url.starts_with("postgresql://") {
            anyhow::bail!("DATABASE_URL must start with sqlite:// or postgres://");
        }

        // Validate rate limiting
        if self.rate_limit.requests_per_second.get() < 1 {
            anyhow::bail!("RATE_LIMIT_RPS must be at least 1");
        }

        if self.rate_limit.burst_size.get() < 1 {
            anyhow::bail!("RATE_LIMIT_BURST must be at least 1");
        }

        // Warn about permissive CORS in production
        if self.security.permissive_cors && std::env::var("RUST_ENV").unwrap_or_default() == "production" {
            tracing::warn!("PERMISSIVE_CORS is enabled in production - this is not recommended!");
        }

        Ok(())
    }
}

impl ServerConfig {
    fn from_env() -> Result<Self> {
        let bind_address = std::env::var("BIND_ADDRESS")
            .unwrap_or_else(|_| "127.0.0.1:8080".to_string())
            .parse()
            .context("Invalid BIND_ADDRESS format")?;

        let max_body_size = std::env::var("MAX_BODY_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2 * 1024 * 1024); // 2MB default

        Ok(Self {
            bind_address,
            max_body_size,
        })
    }
}

impl DatabaseConfig {
    fn from_env() -> Result<Self> {
        let url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "sqlite://data/claims.db".to_string());

        let max_connections = std::env::var("DB_MAX_CONNECTIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        let connect_timeout_seconds = std::env::var("DB_CONNECT_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);

        Ok(Self {
            url,
            max_connections,
            connect_timeout_seconds,
        })
    }
}

impl SecurityConfig {
    fn from_env() -> Result<Self> {
        let allowed_origins_str = std::env::var("ALLOWED_ORIGINS")
            .unwrap_or_else(|_| "http://localhost:3000,http://localhost:8080".to_string());

        let allowed_origins: Vec<String> = allowed_origins_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let permissive_cors = std::env::var("PERMISSIVE_CORS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(false);

        Ok(Self {
            allowed_origins,
            permissive_cors,
        })
    }
}

impl RateLimitConfig {
    fn from_env() -> Result<Self> {
        let requests_per_second = std::env::var("RATE_LIMIT_RPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .and_then(NonZeroU32::new)
            .unwrap_or_else(|| NonZeroU32::new(100).unwrap());

        let burst_size = std::env::var("RATE_LIMIT_BURST")
            .ok()
            .and_then(|s| s.parse().ok())
            .and_then(NonZeroU32::new)
            .unwrap_or_else(|| NonZeroU32::new(20).unwrap());

        Ok(Self {
            requests_per_second,
            burst_size,
        })
    }
}

impl ObservabilityConfig {
    fn from_env() -> Result<Self> {
        let log_level = std::env::var("RUST_LOG")
            .or_else(|_| std::env::var("LOG_LEVEL"))
            .unwrap_or_else(|_| "info".to_string());

        let json_logs = std::env::var("JSON_LOGS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(false);

        let request_tracing = std::env::var("REQUEST_TRACING")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(true);

        Ok(Self {
            log_level,
            json_logs,
            request_tracing,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_defaults() {
        std::env::remove_var("BIND_ADDRESS");
        std::env::remove_var("MAX_BODY_SIZE");

        let config = ServerConfig::from_env().unwrap();
        assert_eq!(config.bind_address.to_string(), "127.0.0.1:8080");
        assert_eq!(config.max_body_size, 2 * 1024 * 1024);
    }

    #[test]
    fn test_server_config_from_env() {
        std::env::set_var("BIND_ADDRESS", "127.0.0.1:3000");
        std::env::set_var("MAX_BODY_SIZE", "5242880");

        let config = ServerConfig::from_env().unwrap();
        assert_eq!(config.bind_address.to_string(), "127.0.0.1:3000");
        assert_eq!(config.max_body_size, 5242880);

        std::env::remove_var("BIND_ADDRESS");
        std::env::remove_var("MAX_BODY_SIZE");
    }

    #[test]
    fn test_database_config_defaults() {
        std::env::remove_var("DATABASE_URL");
        std::env::remove_var("DB_MAX_CONNECTIONS");
        std::env::remove_var("DB_CONNECT_TIMEOUT");

        let config = DatabaseConfig::from_env().unwrap();
        assert_eq!(config.url, "sqlite://data/claims.db");
        assert_eq!(config.max_connections, 10);
        assert_eq!(config.connect_timeout_seconds, 30);
    }

    #[test]
    fn test_rate_limit_config_defaults() {
        std::env::remove_var("RATE_LIMIT_RPS");
        std::env::remove_var("RATE_LIMIT_BURST");

        let config = RateLimitConfig::from_env().unwrap();
        assert_eq!(config.requests_per_second.get(), 100);
        assert_eq!(config.burst_size.get(), 20);
    }

    #[test]
    fn test_rate_limit_config_from_env() {
        std::env::set_var("RATE_LIMIT_RPS", "50");
        std::env::set_var("RATE_LIMIT_BURST", "10");

        let config = RateLimitConfig::from_env().unwrap();
        assert_eq!(config.requests_per_second.get(), 50);
        assert_eq!(config.burst_size.get(), 10);

        std::env::remove_var("RATE_LIMIT_RPS");
        std::env::remove_var("RATE_LIMIT_BURST");
    }

    #[test]
    fn test_security_config_defaults() {
        std::env::remove_var("ALLOWED_ORIGINS");
        std::env::remove_var("PERMISSIVE_CORS");

        let config = SecurityConfig::from_env().unwrap();
        assert_eq!(config.allowed_origins.len(), 2);
        assert!(!config.permissive_cors);
    }

    #[test]
    fn test_observability_config_defaults() {
        std::env::remove_var("RUST_LOG");
        std::env::remove_var("LOG_LEVEL");
        std::env::remove_var("JSON_LOGS");
        std::env::remove_var("REQUEST_TRACING");

        let config = ObservabilityConfig::from_env().unwrap();
        assert_eq!(config.log_level, "info");
        assert!(!config.json_logs);
        assert!(config.request_tracing);
    }

    #[test]
    fn test_config_validation_max_body_size() {
        std::env::set_var("MAX_BODY_SIZE", "200000000"); // 200MB

        let result = Config::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("MAX_BODY_SIZE too large"));

        std::env::remove_var("MAX_BODY_SIZE");
    }

    #[test]
    fn test_config_validation_database_url() {
        std::env::set_var("DATABASE_URL", "invalid://url");

        let result = Config::from_env();
        assert!(result.is_err());

        std::env::remove_var("DATABASE_URL");
    }
}
