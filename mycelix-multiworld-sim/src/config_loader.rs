// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Configuration loader: reads TOML config files for disasters, tech tree, and policy.
//!
//! Falls back to hardcoded defaults if files are missing or malformed.
//! This enables data-driven parameter tuning without recompilation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// DISASTER CONFIG
// ============================================================================

/// Top-level disaster configuration from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisasterConfig {
    #[serde(default)]
    pub solar: CategoryConfig,
    #[serde(default)]
    pub impact: CategoryConfig,
    #[serde(default)]
    pub planetary: CategoryConfig,
    #[serde(default)]
    pub eclss: CategoryConfig,
    #[serde(default)]
    pub psychological: CategoryConfig,
    #[serde(default)]
    pub technology: CategoryConfig,
    #[serde(default)]
    pub civilization: CategoryConfig,
}

/// A disaster category with enable toggle and per-event probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Event-specific overrides. Key = event name, value = probability override.
    #[serde(flatten)]
    pub events: HashMap<String, toml::Value>,
}

impl Default for CategoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            events: HashMap::new(),
        }
    }
}

/// Extract a probability from the TOML event config, or return the default.
pub fn event_probability(config: &CategoryConfig, event_name: &str, default: f64) -> f64 {
    if !config.enabled {
        return 0.0;
    }
    config
        .events
        .get(event_name)
        .and_then(|v| v.as_table())
        .and_then(|t| t.get("probability"))
        .and_then(|p| p.as_float())
        .unwrap_or(default)
}

// ============================================================================
// TECH TREE CONFIG
// ============================================================================

/// Tech tree configuration from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechTreeConfig {
    #[serde(default)]
    pub milestones: HashMap<String, MilestoneConfig>,
}

/// A single tech milestone configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilestoneConfig {
    pub name: String,
    #[serde(default)]
    pub earliest_tick: u32,
    #[serde(default)]
    pub latest_tick: u32,
    #[serde(default)]
    pub base_probability: f64,
    #[serde(default)]
    pub prerequisite_sectors: Vec<Vec<f64>>,
    #[serde(default)]
    pub prerequisite_milestones: Vec<String>,
    #[serde(default)]
    pub effects: HashMap<String, toml::Value>,
}

// ============================================================================
// LOADING FUNCTIONS
// ============================================================================

/// Load disaster config from TOML file. Falls back to empty (all defaults) on error.
pub fn load_disaster_config(path: &Path) -> DisasterConfig {
    match std::fs::read_to_string(path) {
        Ok(content) => match toml::from_str(&content) {
            Ok(config) => {
                tracing::info!("Loaded disaster config from {}", path.display());
                config
            }
            Err(e) => {
                tracing::warn!("Failed to parse {}: {}. Using defaults.", path.display(), e);
                DisasterConfig::default()
            }
        },
        Err(_) => {
            tracing::info!(
                "No disaster config at {}. Using hardcoded defaults.",
                path.display()
            );
            DisasterConfig::default()
        }
    }
}

/// Load tech tree config from TOML file. Falls back to empty on error.
pub fn load_tech_tree_config(path: &Path) -> TechTreeConfig {
    match std::fs::read_to_string(path) {
        Ok(content) => match toml::from_str(&content) {
            Ok(config) => {
                tracing::info!("Loaded tech tree from {}", path.display());
                config
            }
            Err(e) => {
                tracing::warn!("Failed to parse {}: {}. Using defaults.", path.display(), e);
                TechTreeConfig {
                    milestones: HashMap::new(),
                }
            }
        },
        Err(_) => {
            tracing::info!(
                "No tech tree config at {}. Using hardcoded defaults.",
                path.display()
            );
            TechTreeConfig {
                milestones: HashMap::new(),
            }
        }
    }
}

impl Default for DisasterConfig {
    fn default() -> Self {
        Self {
            solar: CategoryConfig::default(),
            impact: CategoryConfig::default(),
            planetary: CategoryConfig::default(),
            eclss: CategoryConfig::default(),
            psychological: CategoryConfig::default(),
            technology: CategoryConfig::default(),
            civilization: CategoryConfig::default(),
        }
    }
}

fn default_true() -> bool {
    true
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_disaster_config_all_enabled() {
        let config = DisasterConfig::default();
        assert!(config.solar.enabled);
        assert!(config.eclss.enabled);
        assert!(config.civilization.enabled);
    }

    #[test]
    fn load_nonexistent_file_returns_default() {
        let config = load_disaster_config(Path::new("/nonexistent/path.toml"));
        assert!(config.solar.enabled);
    }

    #[test]
    fn parse_disasters_toml() {
        let toml_str = r#"
[solar]
enabled = true

[solar.carrington_event]
probability = 0.001

[eclss]
enabled = false
"#;
        let config: DisasterConfig = toml::from_str(toml_str).unwrap();
        assert!(config.solar.enabled);
        assert!(!config.eclss.enabled);
        let p = event_probability(&config.solar, "carrington_event", 0.00058);
        assert!((p - 0.001).abs() < 1e-6, "Override should work: {}", p);
    }

    #[test]
    fn event_probability_falls_back_to_default() {
        let config = CategoryConfig::default();
        let p = event_probability(&config, "nonexistent_event", 0.05);
        assert!((p - 0.05).abs() < 1e-10);
    }

    #[test]
    fn event_probability_zero_when_disabled() {
        let config = CategoryConfig {
            enabled: false,
            events: HashMap::new(),
        };
        let p = event_probability(&config, "anything", 0.99);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn parse_tech_tree_toml() {
        let toml_str = r#"
[milestones.fusion_demo]
name = "Fusion Demo"
earliest_tick = 120
latest_tick = 240
base_probability = 0.004
prerequisite_sectors = [[0, 2.5], [4, 2.0]]
prerequisite_milestones = ["fission_surface_power"]
"#;
        let config: TechTreeConfig = toml::from_str(toml_str).unwrap();
        assert!(config.milestones.contains_key("fusion_demo"));
        let m = &config.milestones["fusion_demo"];
        assert_eq!(m.name, "Fusion Demo");
        assert_eq!(m.earliest_tick, 120);
        assert!((m.base_probability - 0.004).abs() < 1e-10);
    }

    #[test]
    fn load_real_disasters_toml() {
        let path = Path::new("config/disasters.toml");
        if path.exists() {
            let config = load_disaster_config(path);
            assert!(config.solar.enabled);
            let p = event_probability(&config.solar, "carrington_event", 0.0);
            assert!(p > 0.0, "Carrington should have nonzero probability: {}", p);
        }
    }

    #[test]
    fn load_real_tech_tree_toml() {
        let path = Path::new("config/tech_tree.toml");
        if path.exists() {
            let config = load_tech_tree_config(path);
            assert!(!config.milestones.is_empty(), "Should have milestones");
        }
    }
}
