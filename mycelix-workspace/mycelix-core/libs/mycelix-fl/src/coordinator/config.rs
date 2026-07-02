// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Configuration for the FL coordinator.

use serde::{Deserialize, Serialize};

/// Configuration parameters for the FL coordinator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// Minimum nodes required to start a round (default 3).
    pub min_nodes: usize,

    /// Maximum registered nodes (default 1000).
    pub max_nodes: usize,

    /// Timeout in seconds for gradient collection per round (default 300).
    pub round_timeout_secs: u64,

    /// Maximum number of FL rounds. `None` means unlimited.
    pub max_rounds: Option<u64>,

    /// Whether nodes must present credentials to submit gradients (default true).
    pub require_authentication: bool,

    /// Maximum gradient submissions per node per minute (default 10).
    pub rate_limit_per_minute: usize,

    /// Maximum gradient submissions per node per round (default 1).
    pub rate_limit_per_round: usize,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            min_nodes: 3,
            max_nodes: 1000,
            round_timeout_secs: 300,
            max_rounds: None,
            require_authentication: true,
            rate_limit_per_minute: 10,
            rate_limit_per_round: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = CoordinatorConfig::default();
        assert_eq!(cfg.min_nodes, 3);
        assert_eq!(cfg.max_nodes, 1000);
        assert_eq!(cfg.round_timeout_secs, 300);
        assert!(cfg.max_rounds.is_none());
        assert!(cfg.require_authentication);
        assert_eq!(cfg.rate_limit_per_minute, 10);
        assert_eq!(cfg.rate_limit_per_round, 1);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = CoordinatorConfig {
            min_nodes: 5,
            max_nodes: 50,
            round_timeout_secs: 60,
            max_rounds: Some(100),
            require_authentication: false,
            rate_limit_per_minute: 20,
            rate_limit_per_round: 2,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: CoordinatorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.min_nodes, cfg2.min_nodes);
        assert_eq!(cfg.max_nodes, cfg2.max_nodes);
        assert_eq!(cfg.round_timeout_secs, cfg2.round_timeout_secs);
        assert_eq!(cfg.max_rounds, cfg2.max_rounds);
        assert_eq!(cfg.require_authentication, cfg2.require_authentication);
    }
}
