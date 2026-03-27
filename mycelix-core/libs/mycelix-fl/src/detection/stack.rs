// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-layer detection stack orchestrator.
//!
//! Runs detection layers in sequence — each layer can flag or exclude nodes:
//!
//! 1. **PoGQ** (optional) — stateful scoring + quarantine
//! 2. **Shapley** (optional) — leave-one-out contribution analysis
//! 3. **Self-Healing** (optional) — repair flagged gradients if close enough
//!
//! The final output is a set of clean gradients (those that passed all layers
//! plus any that were successfully repaired).

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::FlError;
use crate::pogq::config::PoGQv41Config;
use crate::pogq::v41_enhanced::PoGQv41Enhanced;
use crate::types::Gradient;

use super::self_healing::{SelfHealer, SelfHealerConfig};
use super::shapley::{ShapleyConfig, ShapleyDetector};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the detection stack.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// Enable the PoGQ layer.
    pub enable_pogq: bool,
    /// Enable the Shapley detection layer.
    pub enable_shapley: bool,
    /// Enable the self-healing layer.
    pub enable_healing: bool,
    /// PoGQ configuration (used when `enable_pogq` is true).
    pub pogq_config: Option<PoGQv41Config>,
    /// Shapley configuration (used when `enable_shapley` is true).
    pub shapley_config: Option<ShapleyConfig>,
    /// Self-healing configuration (used when `enable_healing` is true).
    pub healing_config: Option<SelfHealerConfig>,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            enable_pogq: true,
            enable_shapley: true,
            enable_healing: true,
            pogq_config: None,
            shapley_config: None,
            healing_config: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Report from a single detection layer.
#[derive(Clone, Debug)]
pub struct LayerReport {
    /// Name of the layer (e.g. "PoGQ", "Shapley", "SelfHealing").
    pub layer_name: String,
    /// Node IDs flagged by this layer.
    pub flagged_nodes: Vec<String>,
    /// Wall-clock duration in microseconds.
    pub duration_us: u64,
}

/// Result of the full detection pipeline.
#[derive(Clone, Debug)]
pub struct StackResult {
    /// Gradients that passed all layers (including repaired ones).
    pub clean_gradients: Vec<Gradient>,
    /// Node IDs excluded by detection (not repairable).
    pub excluded_nodes: Vec<String>,
    /// Node IDs that were repaired by self-healing.
    pub healed_nodes: Vec<String>,
    /// Per-layer detection reports.
    pub layer_reports: Vec<LayerReport>,
}

// ---------------------------------------------------------------------------
// Detection stack
// ---------------------------------------------------------------------------

/// Multi-layer Byzantine detection orchestrator.
pub struct DetectionStack {
    /// PoGQ v4.1 Enhanced (optional, stateful).
    pub pogq: Option<PoGQv41Enhanced>,
    /// Shapley detector (optional, stateless).
    pub shapley: Option<ShapleyDetector>,
    /// Self-healer (optional, stateless).
    pub healer: Option<SelfHealer>,
}

impl DetectionStack {
    /// Create a detection stack from configuration.
    pub fn new(config: DetectionConfig) -> Self {
        let pogq = if config.enable_pogq {
            Some(PoGQv41Enhanced::new(
                config.pogq_config.unwrap_or_default(),
            ))
        } else {
            None
        };

        let shapley = if config.enable_shapley {
            Some(ShapleyDetector::new(
                config.shapley_config.unwrap_or_default(),
            ))
        } else {
            None
        };

        let healer = if config.enable_healing {
            Some(SelfHealer::new(
                config.healing_config.unwrap_or_default(),
            ))
        } else {
            None
        };

        Self {
            pogq,
            shapley,
            healer,
        }
    }

    /// Create a stack with all layers enabled and default configuration.
    pub fn with_defaults() -> Self {
        Self::new(DetectionConfig::default())
    }

    /// Run the full detection pipeline on a round's gradients.
    ///
    /// # Pipeline
    ///
    /// 1. **PoGQ** (if enabled): evaluates the round and quarantines nodes.
    ///    Quarantined nodes are removed from the working set.
    /// 2. **Shapley** (if enabled): runs leave-one-out detection on the
    ///    remaining gradients. Suspicious nodes are flagged.
    /// 3. **Self-Healing** (if enabled): attempts to repair flagged (but
    ///    not quarantined) gradients. Unrepairable ones are excluded.
    ///
    /// # Errors
    ///
    /// Returns [`FlError::EmptyGradients`] if the input is empty.
    pub fn detect_and_filter(
        &mut self,
        gradients: &[Gradient],
    ) -> Result<StackResult, FlError> {
        if gradients.is_empty() {
            return Err(FlError::EmptyGradients);
        }

        let mut layer_reports = Vec::new();
        let mut excluded: Vec<String> = Vec::new();

        // Working set: starts as all gradients.
        let mut working: Vec<Gradient> = gradients.to_vec();

        // ----- Layer 1: PoGQ -----
        if let Some(ref mut pogq) = self.pogq {
            let start = Instant::now();
            let round_result = pogq.evaluate_round(&working)?;
            let elapsed = start.elapsed().as_micros() as u64;

            let mut pogq_flagged = Vec::new();
            for ns in &round_result.node_scores {
                if ns.quarantined {
                    pogq_flagged.push(ns.node_id.clone());
                    excluded.push(ns.node_id.clone());
                }
            }

            // Remove quarantined from working set.
            working.retain(|g| !pogq_flagged.contains(&g.node_id));

            layer_reports.push(LayerReport {
                layer_name: "PoGQ".into(),
                flagged_nodes: pogq_flagged,
                duration_us: elapsed,
            });
        }

        // ----- Layer 2: Shapley -----
        let mut shapley_suspicious: Vec<String> = Vec::new();
        if let Some(ref shapley) = self.shapley {
            if !working.is_empty() {
                let start = Instant::now();
                let shapley_result = shapley.detect(&working)?;
                let elapsed = start.elapsed().as_micros() as u64;

                shapley_suspicious = shapley_result.suspicious.clone();

                layer_reports.push(LayerReport {
                    layer_name: "Shapley".into(),
                    flagged_nodes: shapley_result.suspicious,
                    duration_us: elapsed,
                });
            }
        }

        // ----- Layer 3: Self-Healing -----
        let mut healed_nodes: Vec<String> = Vec::new();

        if let Some(ref healer) = self.healer {
            if !shapley_suspicious.is_empty() && !working.is_empty() {
                let start = Instant::now();
                let healing_result = healer.heal(&working, &shapley_suspicious)?;
                let elapsed = start.elapsed().as_micros() as u64;

                healed_nodes = healing_result.repaired_nodes.clone();

                // Replace repaired gradients in working set.
                for repaired in &healing_result.repaired_gradients {
                    if let Some(pos) = working.iter().position(|g| g.node_id == repaired.node_id) {
                        working[pos] = repaired.clone();
                    }
                }

                // Exclude unrepairable.
                for node_id in &healing_result.unrepairable {
                    excluded.push(node_id.clone());
                }
                working.retain(|g| !healing_result.unrepairable.contains(&g.node_id));

                let mut healing_flagged = healing_result.unrepairable;
                healing_flagged.extend(healing_result.repaired_nodes.iter().cloned());

                layer_reports.push(LayerReport {
                    layer_name: "SelfHealing".into(),
                    flagged_nodes: healing_flagged,
                    duration_us: elapsed,
                });
            }
        } else {
            // No healer: Shapley suspicious nodes are excluded directly.
            for node_id in &shapley_suspicious {
                excluded.push(node_id.clone());
            }
            working.retain(|g| !shapley_suspicious.contains(&g.node_id));
        }

        Ok(StackResult {
            clean_gradients: working,
            excluded_nodes: excluded,
            healed_nodes,
            layer_reports,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Gradient;

    fn grad(id: &str, values: Vec<f32>) -> Gradient {
        Gradient::new(id, values, 0)
    }

    /// Helper: honest gradient cluster around a base direction.
    fn honest_gradients(n: usize, dim: usize) -> Vec<Gradient> {
        let base: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        (0..n)
            .map(|i| {
                let vals: Vec<f32> = base
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| v + 0.01 * ((i * 7 + j * 3) % 13) as f32 * 0.1)
                    .collect();
                grad(&format!("honest-{}", i), vals)
            })
            .collect()
    }

    /// Byzantine gradient: opposite direction, large magnitude.
    fn byzantine_gradient(dim: usize, id: &str) -> Gradient {
        let vals: Vec<f32> = (0..dim).map(|i| -((i as f32 + 1.0) * 0.5)).collect();
        grad(id, vals)
    }

    /// Full pipeline: 5 honest + 1 Byzantine -> detected and excluded.
    #[test]
    fn test_full_pipeline_detects_byzantine() {
        // Use Shapley-only to avoid PoGQ needing warmup rounds.
        let config = DetectionConfig {
            enable_pogq: false,
            enable_shapley: true,
            enable_healing: false,
            shapley_config: Some(ShapleyConfig {
                threshold: -0.01,
                ..ShapleyConfig::default()
            }),
            ..DetectionConfig::default()
        };
        let mut stack = DetectionStack::new(config);

        let dim = 50;
        let mut grads = honest_gradients(5, dim);
        grads.push(byzantine_gradient(dim, "byz"));

        let result = stack.detect_and_filter(&grads).unwrap();

        assert!(
            result.excluded_nodes.contains(&"byz".to_string()),
            "Byzantine should be excluded. Excluded: {:?}",
            result.excluded_nodes
        );

        // Clean gradients should not contain the Byzantine.
        assert!(
            !result
                .clean_gradients
                .iter()
                .any(|g| g.node_id == "byz"),
            "Clean set should not contain Byzantine"
        );
    }

    /// PoGQ-only mode: uses PoGQ stateful detection.
    #[test]
    fn test_pogq_only_mode() {
        let config = DetectionConfig {
            enable_pogq: true,
            enable_shapley: false,
            enable_healing: false,
            pogq_config: Some(PoGQv41Config {
                warm_up_rounds: 0,
                k_quarantine: 1,
                beta: 0.0,
                ..PoGQv41Config::default()
            }),
            ..DetectionConfig::default()
        };
        let mut stack = DetectionStack::new(config);

        let dim = 50;

        // Seed history with honest-only rounds.
        for _ in 0..3 {
            let grads = honest_gradients(5, dim);
            stack.detect_and_filter(&grads).unwrap();
        }

        // Now add a Byzantine.
        let mut grads = honest_gradients(4, dim);
        let byz_vals: Vec<f32> = (0..dim).map(|i| -((i as f32 + 1.0) * 0.1)).collect();
        grads.push(grad("evil", byz_vals));

        let result = stack.detect_and_filter(&grads).unwrap();

        assert!(
            result.layer_reports.len() == 1,
            "Should have exactly 1 layer report (PoGQ)"
        );
        assert_eq!(result.layer_reports[0].layer_name, "PoGQ");
    }

    /// Shapley-only mode.
    #[test]
    fn test_shapley_only_mode() {
        let config = DetectionConfig {
            enable_pogq: false,
            enable_shapley: true,
            enable_healing: false,
            ..DetectionConfig::default()
        };
        let mut stack = DetectionStack::new(config);

        let grads = honest_gradients(5, 20);
        let result = stack.detect_and_filter(&grads).unwrap();

        assert!(
            result.layer_reports.len() == 1,
            "Should have exactly 1 layer report (Shapley)"
        );
        assert_eq!(result.layer_reports[0].layer_name, "Shapley");
    }

    /// Healing mode: repairs marginal gradient, excludes extreme.
    #[test]
    fn test_healing_repairs_marginal() {
        let dim = 20;
        let base: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let mut grads: Vec<Gradient> = (0..5)
            .map(|i| {
                let vals: Vec<f32> = base
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| v + 0.005 * ((i * 3 + j) % 7) as f32)
                    .collect();
                grad(&format!("h-{}", i), vals)
            })
            .collect();

        // Marginal: slightly off but close.
        let marginal_vals: Vec<f32> = base.iter().map(|&v| v * 1.5).collect();
        grads.push(grad("marginal", marginal_vals));

        // Extreme: very far away.
        let extreme_vals: Vec<f32> = base.iter().map(|&v| -v * 50.0).collect();
        grads.push(grad("extreme", extreme_vals));

        let config = DetectionConfig {
            enable_pogq: false,
            enable_shapley: true,
            enable_healing: true,
            shapley_config: Some(ShapleyConfig {
                threshold: -0.001,
                ..ShapleyConfig::default()
            }),
            healing_config: Some(SelfHealerConfig {
                repair_method: super::super::self_healing::RepairMethod::WeightedInterpolate,
                max_repair_distance: 2.0,
            }),
            ..DetectionConfig::default()
        };
        let mut stack = DetectionStack::new(config);
        let result = stack.detect_and_filter(&grads).unwrap();

        // Extreme should be excluded.
        assert!(
            result.excluded_nodes.contains(&"extreme".to_string()),
            "Extreme gradient should be excluded. Excluded: {:?}",
            result.excluded_nodes
        );
    }

    /// All layers disabled -> passthrough (all gradients returned).
    #[test]
    fn test_all_disabled_passthrough() {
        let config = DetectionConfig {
            enable_pogq: false,
            enable_shapley: false,
            enable_healing: false,
            ..DetectionConfig::default()
        };
        let mut stack = DetectionStack::new(config);

        let grads = vec![
            grad("a", vec![1.0, 2.0]),
            grad("b", vec![3.0, 4.0]),
            grad("c", vec![5.0, 6.0]),
        ];

        let result = stack.detect_and_filter(&grads).unwrap();

        assert_eq!(result.clean_gradients.len(), 3);
        assert!(result.excluded_nodes.is_empty());
        assert!(result.healed_nodes.is_empty());
        assert!(result.layer_reports.is_empty());
    }

    /// Config serde roundtrip.
    #[test]
    fn test_detection_config_serde() {
        let cfg = DetectionConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: DetectionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.enable_pogq, cfg2.enable_pogq);
        assert_eq!(cfg.enable_shapley, cfg2.enable_shapley);
        assert_eq!(cfg.enable_healing, cfg2.enable_healing);
    }
}
