// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bridge configuration — loaded from TOML or environment.

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BridgeConfig {
    pub conductor_url: String,
    pub app_id: String,
    pub accounts: Vec<AccountConfig>,
    /// OAuth2 client ID (from GCP / Azure AD)
    pub oauth_client_id: Option<String>,
    /// OAuth2 client secret
    pub oauth_client_secret: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccountConfig {
    pub email: String,
    pub provider: String,
    pub imap_host: String,
    pub imap_port: u16,
    pub imap_tls: bool,
    pub smtp_host: String,
    pub smtp_port: u16,
    pub smtp_tls: bool,
    pub username: String,
    pub password: String,
    pub enabled: bool,
    /// Polling interval in seconds
    pub sync_interval_secs: u32,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            conductor_url: "ws://127.0.0.1:8888".to_string(),
            app_id: "mycelix_mail".to_string(),
            accounts: vec![],
            oauth_client_id: None,
            oauth_client_secret: None,
        }
    }
}

pub fn load_config() -> Result<BridgeConfig> {
    let config_path = std::env::var("MAIL_BRIDGE_CONFIG")
        .unwrap_or_else(|_| "bridge-config.toml".to_string());

    if std::path::Path::new(&config_path).exists() {
        let content = std::fs::read_to_string(&config_path)?;
        let config: BridgeConfig = toml::from_str(&content)?;
        Ok(config)
    } else {
        tracing::warn!("No config file found at {config_path}, using defaults");
        Ok(BridgeConfig::default())
    }
}
