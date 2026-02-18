//! Consciousness-Aware Byzantine Plugin
//!
//! Uses Phi (integrated information) scores to adjust FL aggregation weights.
//! Nodes with low consciousness integration get dampened or vetoed; nodes with
//! high Phi get boosted. This closes the consciousness loop in federated learning.
//!
//! # Usage
//!
//! ```ignore
//! let mut plugin = ConsciousnessAwareByzantinePlugin::new();
//! // Set phi scores from Symthaea before each round
//! plugin.set_phi_scores(scores);
//! // Use with PipelinePlugins
//! let mut plugins = PipelinePlugins {
//!     byzantine: vec![&mut plugin],
//!     ..PipelinePlugins::none()
//! };
//! ```

use std::collections::HashMap;

use crate::pipeline::{ExternalWeightMap, ParticipantWeightAdjustment};
use crate::plugins::ByzantinePlugin;
use crate::types::GradientUpdate;

// Canonical Phi threshold constants — must match mycelix_bridge_common::phi_thresholds::PhiThresholds::default().
// Source of truth: crates/mycelix-bridge-common/src/phi_thresholds.rs
const CANONICAL_FL_VETO: f32 = 0.1;
const CANONICAL_FL_DAMPEN: f32 = 0.3;
const CANONICAL_FL_BOOST: f32 = 0.6;
const CANONICAL_FL_DAMPEN_FACTOR: f32 = 0.3;
const CANONICAL_FL_BOOST_FACTOR: f32 = 1.5;

/// Configuration for consciousness-aware Byzantine detection.
///
/// Default values are aligned with the canonical thresholds in
/// `mycelix_bridge_common::phi_thresholds::PhiThresholds`. If you
/// need to override per-instance, construct with custom values;
/// otherwise prefer `Default::default()` to stay aligned.
#[derive(Debug, Clone)]
pub struct ConsciousnessConfig {
    /// Below this phi: dampen weight (default 0.3, canonical: fl_dampen)
    pub phi_threshold: f32,
    /// Above this phi: boost weight (default 0.6, canonical: fl_boost)
    pub phi_boost_threshold: f32,
    /// Weight multiplier for low-phi participants (default 0.3, canonical: fl_dampen_factor)
    pub dampen_factor: f32,
    /// Weight multiplier for high-phi participants (default 1.5, canonical: fl_boost_factor)
    pub boost_factor: f32,
    /// Below this phi: veto entirely (default 0.1, canonical: fl_veto)
    pub veto_threshold: f32,
    /// Default phi for participants without a score (default 0.5 = neutral)
    pub default_phi: f32,
}

impl Default for ConsciousnessConfig {
    fn default() -> Self {
        Self {
            phi_threshold: CANONICAL_FL_DAMPEN,
            phi_boost_threshold: CANONICAL_FL_BOOST,
            dampen_factor: CANONICAL_FL_DAMPEN_FACTOR,
            boost_factor: CANONICAL_FL_BOOST_FACTOR,
            veto_threshold: CANONICAL_FL_VETO,
            default_phi: 0.5,
        }
    }
}

/// Consciousness-aware Byzantine detection plugin.
///
/// Maps per-participant Phi scores to weight adjustments in the FL pipeline.
/// Phi scores must be set externally each round via [`set_phi_scores`].
pub struct ConsciousnessAwareByzantinePlugin {
    config: ConsciousnessConfig,
    /// Per-participant phi scores, set externally before each round.
    phi_scores: HashMap<String, f32>,
}

impl ConsciousnessAwareByzantinePlugin {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self::with_config(ConsciousnessConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: ConsciousnessConfig) -> Self {
        Self {
            config,
            phi_scores: HashMap::new(),
        }
    }

    /// Set phi scores for all participants in the current round.
    pub fn set_phi_scores(&mut self, scores: HashMap<String, f32>) {
        self.phi_scores = scores;
    }

    /// Get the phi score for a participant (or default if missing).
    pub fn phi_for(&self, participant_id: &str) -> f32 {
        self.phi_scores
            .get(participant_id)
            .copied()
            .unwrap_or(self.config.default_phi)
    }

    /// Get the current configuration.
    pub fn config(&self) -> &ConsciousnessConfig {
        &self.config
    }
}

impl Default for ConsciousnessAwareByzantinePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl ByzantinePlugin for ConsciousnessAwareByzantinePlugin {
    fn analyze(&mut self, updates: &[GradientUpdate]) -> ExternalWeightMap {
        let mut weights = ExternalWeightMap::new();

        for update in updates {
            let phi = self.phi_for(&update.participant_id);

            if phi < self.config.veto_threshold {
                // Extremely low phi: veto entirely
                weights.insert(
                    update.participant_id.clone(),
                    vec![ParticipantWeightAdjustment {
                        weight_multiplier: 0.0,
                        veto: true,
                        source: "consciousness_aware".into(),
                    }],
                );
            } else if phi < self.config.phi_threshold {
                // Low phi: dampen
                weights.insert(
                    update.participant_id.clone(),
                    vec![ParticipantWeightAdjustment {
                        weight_multiplier: self.config.dampen_factor,
                        veto: false,
                        source: "consciousness_aware".into(),
                    }],
                );
            } else if phi > self.config.phi_boost_threshold {
                // High phi: boost
                weights.insert(
                    update.participant_id.clone(),
                    vec![ParticipantWeightAdjustment {
                        weight_multiplier: self.config.boost_factor,
                        veto: false,
                        source: "consciousness_aware".into(),
                    }],
                );
            }
            // else: neutral (phi between threshold and boost_threshold), no adjustment
        }

        weights
    }

    fn name(&self) -> &str {
        "consciousness_aware"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_update(id: &str) -> GradientUpdate {
        GradientUpdate::new(id.into(), 1, vec![0.5; 10], 100, 0.5)
    }

    #[test]
    fn test_default_config() {
        let config = ConsciousnessConfig::default();
        assert_eq!(config.phi_threshold, 0.3);
        assert_eq!(config.phi_boost_threshold, 0.6);
        assert_eq!(config.dampen_factor, 0.3);
        assert_eq!(config.boost_factor, 1.5);
        assert_eq!(config.veto_threshold, 0.1);
        assert_eq!(config.default_phi, 0.5);
    }

    #[test]
    fn test_custom_config() {
        let config = ConsciousnessConfig {
            phi_threshold: 0.4,
            phi_boost_threshold: 0.8,
            dampen_factor: 0.1,
            boost_factor: 2.0,
            veto_threshold: 0.05,
            default_phi: 0.3,
        };
        let plugin = ConsciousnessAwareByzantinePlugin::with_config(config.clone());
        assert_eq!(plugin.config().phi_threshold, 0.4);
        assert_eq!(plugin.config().boost_factor, 2.0);
    }

    #[test]
    fn test_high_phi_gets_boosted() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        let mut scores = HashMap::new();
        scores.insert("node_a".to_string(), 0.8);
        plugin.set_phi_scores(scores);

        let updates = vec![make_update("node_a")];
        let weights = plugin.analyze(&updates);

        assert!(weights.contains_key("node_a"));
        let adj = &weights["node_a"][0];
        assert_eq!(adj.weight_multiplier, 1.5);
        assert!(!adj.veto);
    }

    #[test]
    fn test_low_phi_gets_dampened() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        let mut scores = HashMap::new();
        scores.insert("node_b".to_string(), 0.2);
        plugin.set_phi_scores(scores);

        let updates = vec![make_update("node_b")];
        let weights = plugin.analyze(&updates);

        assert!(weights.contains_key("node_b"));
        let adj = &weights["node_b"][0];
        assert_eq!(adj.weight_multiplier, 0.3);
        assert!(!adj.veto);
    }

    #[test]
    fn test_very_low_phi_gets_vetoed() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        let mut scores = HashMap::new();
        scores.insert("node_c".to_string(), 0.05);
        plugin.set_phi_scores(scores);

        let updates = vec![make_update("node_c")];
        let weights = plugin.analyze(&updates);

        assert!(weights.contains_key("node_c"));
        let adj = &weights["node_c"][0];
        assert_eq!(adj.weight_multiplier, 0.0);
        assert!(adj.veto);
    }

    #[test]
    fn test_neutral_phi_no_adjustment() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        let mut scores = HashMap::new();
        scores.insert("node_d".to_string(), 0.45); // Between 0.3 and 0.6
        plugin.set_phi_scores(scores);

        let updates = vec![make_update("node_d")];
        let weights = plugin.analyze(&updates);

        // Neutral range: no entry in weight map
        assert!(
            !weights.contains_key("node_d"),
            "Neutral phi should produce no adjustment"
        );
    }

    #[test]
    fn test_missing_participant_uses_default() {
        let plugin = ConsciousnessAwareByzantinePlugin::new();
        // default_phi = 0.5, which is in the neutral range (0.3..0.6)
        assert_eq!(plugin.phi_for("unknown_node"), 0.5);
    }

    #[test]
    fn test_missing_participant_default_is_neutral() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        // No phi scores set — all participants use default (0.5 = neutral)
        let updates = vec![make_update("node_x"), make_update("node_y")];
        let weights = plugin.analyze(&updates);

        assert!(
            weights.is_empty(),
            "Default phi (0.5) is neutral, should produce no adjustments"
        );
    }

    #[test]
    fn test_mixed_phi_scores() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        let mut scores = HashMap::new();
        scores.insert("high".to_string(), 0.9);
        scores.insert("mid".to_string(), 0.5);
        scores.insert("low".to_string(), 0.2);
        scores.insert("veto".to_string(), 0.05);
        plugin.set_phi_scores(scores);

        let updates = vec![
            make_update("high"),
            make_update("mid"),
            make_update("low"),
            make_update("veto"),
        ];
        let weights = plugin.analyze(&updates);

        // High: boosted
        assert!(weights.contains_key("high"));
        assert_eq!(weights["high"][0].weight_multiplier, 1.5);
        assert!(!weights["high"][0].veto);

        // Mid: neutral (no entry)
        assert!(!weights.contains_key("mid"));

        // Low: dampened
        assert!(weights.contains_key("low"));
        assert_eq!(weights["low"][0].weight_multiplier, 0.3);
        assert!(!weights["low"][0].veto);

        // Veto: vetoed
        assert!(weights.contains_key("veto"));
        assert!(weights["veto"][0].veto);
    }

    #[test]
    fn test_boundary_values() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        let mut scores = HashMap::new();
        // Exact boundary values
        scores.insert("at_veto".to_string(), 0.1); // == veto_threshold → dampened (not vetoed)
        scores.insert("at_threshold".to_string(), 0.3); // == phi_threshold → neutral
        scores.insert("at_boost".to_string(), 0.6); // == phi_boost_threshold → neutral
        plugin.set_phi_scores(scores);

        let updates = vec![
            make_update("at_veto"),
            make_update("at_threshold"),
            make_update("at_boost"),
        ];
        let weights = plugin.analyze(&updates);

        // at_veto (0.1): >= veto_threshold(0.1), < phi_threshold(0.3) → dampened
        assert!(weights.contains_key("at_veto"));
        assert!(!weights["at_veto"][0].veto);
        assert_eq!(weights["at_veto"][0].weight_multiplier, 0.3);

        // at_threshold (0.3): >= phi_threshold(0.3), <= phi_boost_threshold(0.6) → neutral
        assert!(!weights.contains_key("at_threshold"));

        // at_boost (0.6): == phi_boost_threshold(0.6), not > → neutral
        assert!(!weights.contains_key("at_boost"));
    }

    #[test]
    fn test_plugin_name() {
        let plugin = ConsciousnessAwareByzantinePlugin::new();
        assert_eq!(ByzantinePlugin::name(&plugin), "consciousness_aware");
    }

    #[test]
    fn test_empty_updates() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        let weights = plugin.analyze(&[]);
        assert!(weights.is_empty());
    }
}
