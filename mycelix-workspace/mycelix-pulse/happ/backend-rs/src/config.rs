// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Configuration management for Mycelix-Mail backend

use std::env;
use std::time::Duration;

/// Application configuration loaded from environment variables
#[derive(Debug, Clone)]
pub struct Config {
    /// Server host (default: 0.0.0.0)
    pub host: String,
    /// Server port (default: 3001)
    pub port: u16,

    /// Holochain conductor WebSocket URL (default: ws://localhost:8888)
    pub holochain_conductor_url: String,
    /// Holochain admin WebSocket URL (default: ws://localhost:8889)
    pub holochain_admin_url: String,
    /// Holochain app ID (default: mycelix_mail)
    pub holochain_app_id: String,
    /// Enable stub mode for testing (default: false)
    pub holochain_stub_mode: bool,
    /// Lair keystore connection URL (optional, for production)
    pub lair_url: Option<String>,
    /// Lair keystore passphrase (optional, for production)
    pub lair_passphrase: Option<String>,
    /// Connection timeout in seconds (default: 30)
    pub holochain_connect_timeout_secs: u64,
    /// Maximum reconnection attempts (default: 5)
    pub holochain_max_reconnect_attempts: u32,

    /// JWT secret key for token signing
    pub jwt_secret: String,
    /// JWT token expiration in hours (default: 24)
    pub jwt_expiration_hours: u64,

    /// Trust score cache TTL in seconds (default: 300 = 5 minutes)
    pub trust_cache_ttl_secs: u64,
    /// Maximum trust cache entries (default: 10000)
    pub trust_cache_max_entries: u64,

    /// Default minimum trust score for inbox filtering (default: 0.3)
    pub default_min_trust: f64,
    /// Byzantine detection threshold (default: 0.2)
    pub byzantine_threshold: f64,

    /// CORS allowed origins (comma-separated)
    pub cors_origins: Vec<String>,

    /// Rate limit: requests per minute per IP (default: 100)
    pub rate_limit_rpm: u32,

    /// Log level (default: info)
    pub log_level: String,

    // Bridge configuration for cross-hApp communication
    /// Bridge WebSocket URL (optional, for production - usually same as conductor URL)
    pub bridge_url: Option<String>,
    /// Enable Bridge stub mode for testing (default: false)
    pub bridge_stub_mode: bool,
    /// Bridge cache TTL in seconds (default: 300 = 5 minutes)
    pub bridge_cache_ttl_secs: u64,
    /// Fallback trust score when Bridge unavailable (default: 0.3)
    pub bridge_fallback_trust: f64,
    /// Bridge coordinator zome name (default: bridge)
    pub bridge_zome_name: String,
    /// Enable cross-hApp reputation integration (default: true)
    /// When enabled, Mail queries Bridge zome for reputation from other Mycelix hApps
    pub bridge_cross_happ_enabled: bool,
    /// Minimum confidence threshold for using cross-hApp reputation (default: 0.3)
    /// Below this confidence, local trust scoring takes precedence
    pub bridge_min_confidence: f64,

    // Identity client configuration
    /// Identity conductor URL (optional, defaults to holochain_conductor_url)
    pub identity_conductor_url: Option<String>,
    /// Enable identity verification on send (default: true)
    pub identity_verify_on_send: bool,
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenvy::dotenv().ok(); // Load .env file if present

        Ok(Self {
            host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: env::var("PORT")
                .unwrap_or_else(|_| "3001".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidValue("PORT must be a number"))?,

            holochain_conductor_url: env::var("HOLOCHAIN_CONDUCTOR_URL")
                .unwrap_or_else(|_| "ws://localhost:8888".to_string()),
            holochain_admin_url: env::var("HOLOCHAIN_ADMIN_URL")
                .unwrap_or_else(|_| "ws://localhost:8889".to_string()),
            holochain_app_id: env::var("HOLOCHAIN_APP_ID")
                .unwrap_or_else(|_| "mycelix_mail".to_string()),
            holochain_stub_mode: env::var("HOLOCHAIN_STUB_MODE")
                .map(|v| v.to_lowercase() == "true" || v == "1")
                .unwrap_or(false),
            lair_url: env::var("LAIR_URL").ok(),
            lair_passphrase: env::var("LAIR_PASSPHRASE").ok(),
            holochain_connect_timeout_secs: env::var("HOLOCHAIN_CONNECT_TIMEOUT_SECS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidValue("HOLOCHAIN_CONNECT_TIMEOUT_SECS must be a number"))?,
            holochain_max_reconnect_attempts: env::var("HOLOCHAIN_MAX_RECONNECT_ATTEMPTS")
                .unwrap_or_else(|_| "5".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidValue("HOLOCHAIN_MAX_RECONNECT_ATTEMPTS must be a number"))?,

            jwt_secret: env::var("JWT_SECRET")
                .map_err(|_| ConfigError::MissingRequired("JWT_SECRET"))?,
            jwt_expiration_hours: env::var("JWT_EXPIRATION_HOURS")
                .unwrap_or_else(|_| "24".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidValue("JWT_EXPIRATION_HOURS must be a number"))?,

            trust_cache_ttl_secs: env::var("TRUST_CACHE_TTL_SECS")
                .unwrap_or_else(|_| "300".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidValue("TRUST_CACHE_TTL_SECS must be a number"))?,
            trust_cache_max_entries: env::var("TRUST_CACHE_MAX_ENTRIES")
                .unwrap_or_else(|_| "10000".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidValue("TRUST_CACHE_MAX_ENTRIES must be a number"))?,

            default_min_trust: env::var("DEFAULT_MIN_TRUST")
                .unwrap_or_else(|_| "0.3".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidValue("DEFAULT_MIN_TRUST must be a number"))?,
            byzantine_threshold: env::var("BYZANTINE_THRESHOLD")
                .unwrap_or_else(|_| "0.2".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidValue("BYZANTINE_THRESHOLD must be a number"))?,

            cors_origins: env::var("CORS_ORIGINS")
                .unwrap_or_else(|_| "http://localhost:5173,http://localhost:3000".to_string())
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),

            rate_limit_rpm: env::var("RATE_LIMIT_RPM")
                .unwrap_or_else(|_| "100".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidValue("RATE_LIMIT_RPM must be a number"))?,

            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),

            // Bridge configuration for cross-hApp reputation
            bridge_url: env::var("BRIDGE_URL").ok(),
            bridge_stub_mode: env::var("BRIDGE_STUB_MODE")
                .map(|v| v.to_lowercase() == "true" || v == "1")
                .unwrap_or(false),
            bridge_cache_ttl_secs: env::var("BRIDGE_CACHE_TTL_SECS")
                .unwrap_or_else(|_| "300".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidValue("BRIDGE_CACHE_TTL_SECS must be a number"))?,
            bridge_fallback_trust: env::var("BRIDGE_FALLBACK_TRUST")
                .unwrap_or_else(|_| "0.3".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidValue("BRIDGE_FALLBACK_TRUST must be a number"))?,
            bridge_zome_name: env::var("BRIDGE_ZOME_NAME")
                .unwrap_or_else(|_| "bridge".to_string()),
            bridge_cross_happ_enabled: env::var("BRIDGE_CROSS_HAPP_ENABLED")
                .map(|v| v.to_lowercase() != "false" && v != "0")
                .unwrap_or(true),
            bridge_min_confidence: env::var("BRIDGE_MIN_CONFIDENCE")
                .unwrap_or_else(|_| "0.3".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidValue("BRIDGE_MIN_CONFIDENCE must be a number"))?,

            // Identity client configuration
            identity_conductor_url: env::var("IDENTITY_CONDUCTOR_URL").ok(),
            identity_verify_on_send: env::var("IDENTITY_VERIFY_ON_SEND")
                .map(|v| v.to_lowercase() != "false" && v != "0")
                .unwrap_or(true),
        })
    }

    /// Get trust cache TTL as Duration
    pub fn trust_cache_ttl(&self) -> Duration {
        Duration::from_secs(self.trust_cache_ttl_secs)
    }

    /// Get JWT expiration as Duration
    pub fn jwt_expiration(&self) -> Duration {
        Duration::from_secs(self.jwt_expiration_hours * 3600)
    }

    /// Get server address
    pub fn server_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Get Holochain connection timeout as Duration
    pub fn holochain_connect_timeout(&self) -> Duration {
        Duration::from_secs(self.holochain_connect_timeout_secs)
    }

    /// Check if stub mode is enabled
    pub fn is_stub_mode(&self) -> bool {
        self.holochain_stub_mode
    }

    /// Get the conductor URL for identity client
    /// Falls back to holochain_conductor_url if not explicitly set
    pub fn conductor_url(&self) -> &str {
        self.identity_conductor_url
            .as_deref()
            .unwrap_or(&self.holochain_conductor_url)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.jwt_secret.len() < 32 {
            return Err(ConfigError::InvalidValue(
                "JWT_SECRET must be at least 32 characters",
            ));
        }
        if self.default_min_trust < 0.0 || self.default_min_trust > 1.0 {
            return Err(ConfigError::InvalidValue("DEFAULT_MIN_TRUST must be between 0.0 and 1.0"));
        }
        if self.byzantine_threshold < 0.0 || self.byzantine_threshold > 1.0 {
            return Err(ConfigError::InvalidValue("BYZANTINE_THRESHOLD must be between 0.0 and 1.0"));
        }
        Ok(())
    }
}

/// Configuration errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Missing required environment variable: {0}")]
    MissingRequired(&'static str),

    #[error("Invalid value for environment variable: {0}")]
    InvalidValue(&'static str),
}
