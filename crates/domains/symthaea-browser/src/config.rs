// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Configuration for the browser agent.

use serde::{Deserialize, Serialize};

use crate::safety::BrowserSafetyPolicy;

/// Configuration for a browser agent instance.
///
/// Observation and timeout limits are enforced by `CdpSession`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserAgentConfig {
    /// CDP debug endpoint (e.g. "http://127.0.0.1:9222" or a WebSocket URL).
    /// If `None`, a new Chrome process will be launched.
    pub cdp_url: Option<String>,
    /// Run Chrome in headless mode.
    pub headless: bool,
    /// Viewport width in pixels.
    pub viewport_width: u32,
    /// Viewport height in pixels.
    pub viewport_height: u32,
    /// Safety policy for URL filtering and Phi thresholds.
    pub safety: BrowserSafetyPolicy,
    /// Maximum number of elements to include in observations.
    /// Limits cognitive load for the HDC encoder.
    pub max_elements: usize,
    /// Navigation timeout in milliseconds.
    pub navigation_timeout_ms: u64,
}

impl Default for BrowserAgentConfig {
    fn default() -> Self {
        Self {
            cdp_url: None,
            headless: true,
            viewport_width: 1280,
            viewport_height: 720,
            safety: BrowserSafetyPolicy::default(),
            max_elements: 128,
            navigation_timeout_ms: 30_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BrowserAgentConfig::default();
        assert!(config.headless);
        assert_eq!(config.viewport_width, 1280);
        assert_eq!(config.viewport_height, 720);
        assert_eq!(config.max_elements, 128);
    }

    #[test]
    fn test_config_serialization() {
        let config = BrowserAgentConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: BrowserAgentConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.viewport_width, config.viewport_width);
    }
}
