//! Consciousness-Aware Byzantine Plugin
//!
//! Uses consciousness scores to adjust FL aggregation weights.
//! Nodes with low consciousness scores get dampened or vetoed; nodes with
//! high scores get boosted. This closes the consciousness loop in federated learning.
//!
//! **Note**: Consciousness scores are currently derived from SpectralConnectivity
//! (Fiedler value), NOT true IIT Phi. See `CONSCIOUSNESS_METRICS.md` for details.
//!
//! # Usage
//!
//! ```ignore
//! let mut plugin = ConsciousnessAwareByzantinePlugin::new();
//! // Set consciousness scores from Symthaea before each round
//! plugin.set_consciousness_scores(scores);
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
use mycelix_bridge_common::consciousness_thresholds::consciousness_thresholds;

/// Configuration for consciousness-aware Byzantine detection.
///
/// Default values are imported from `mycelix_bridge_common::consciousness_thresholds` —
/// the single source of truth for all Mycelix consciousness thresholds. If you
/// need to override per-instance, construct with custom values;
/// otherwise prefer `Default::default()` to stay aligned.
#[derive(Debug, Clone)]
pub struct ConsciousnessConfig {
    /// Below this score: dampen weight (canonical: fl_dampen)
    pub dampen_threshold: f32,
    /// Above this score: boost weight (canonical: fl_boost)
    pub boost_threshold: f32,
    /// Weight multiplier for low-score participants (canonical: fl_dampen_factor)
    pub dampen_factor: f32,
    /// Weight multiplier for high-score participants (canonical: fl_boost_factor)
    pub boost_factor: f32,
    /// Below this score: veto entirely (canonical: fl_veto)
    pub veto_threshold: f32,
    /// Default score for participants without a consciousness score (default 0.5 = neutral)
    pub default_score: f32,
}

impl Default for ConsciousnessConfig {
    fn default() -> Self {
        let t = consciousness_thresholds();
        Self {
            dampen_threshold: t.fl_dampen,
            boost_threshold: t.fl_boost,
            dampen_factor: t.fl_dampen_factor,
            boost_factor: t.fl_boost_factor,
            veto_threshold: t.fl_veto,
            default_score: 0.5,
        }
    }
}

/// Consciousness-aware Byzantine detection plugin.
///
/// Maps per-participant consciousness scores to weight adjustments in the FL pipeline.
/// Scores must be set externally each round via [`set_consciousness_scores`].
pub struct ConsciousnessAwareByzantinePlugin {
    config: ConsciousnessConfig,
    /// Per-participant consciousness scores, set externally before each round.
    consciousness_scores: HashMap<String, f32>,
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
            consciousness_scores: HashMap::new(),
        }
    }

    /// Set consciousness scores for all participants in the current round.
    pub fn set_consciousness_scores(&mut self, scores: HashMap<String, f32>) {
        self.consciousness_scores = scores;
    }

    /// Get the consciousness score for a participant (or default if missing).
    pub fn consciousness_score_for(&self, participant_id: &str) -> f32 {
        self.consciousness_scores
            .get(participant_id)
            .copied()
            .unwrap_or(self.config.default_score)
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
            let score = self.consciousness_score_for(&update.participant_id);

            if score < self.config.veto_threshold {
                // Extremely low consciousness: veto entirely
                weights.insert(
                    update.participant_id.clone(),
                    vec![ParticipantWeightAdjustment {
                        weight_multiplier: 0.0,
                        veto: true,
                        source: "consciousness_aware".into(),
                    }],
                );
            } else if score < self.config.dampen_threshold {
                // Low consciousness: dampen
                weights.insert(
                    update.participant_id.clone(),
                    vec![ParticipantWeightAdjustment {
                        weight_multiplier: self.config.dampen_factor,
                        veto: false,
                        source: "consciousness_aware".into(),
                    }],
                );
            } else if score > self.config.boost_threshold {
                // High consciousness: boost
                weights.insert(
                    update.participant_id.clone(),
                    vec![ParticipantWeightAdjustment {
                        weight_multiplier: self.config.boost_factor,
                        veto: false,
                        source: "consciousness_aware".into(),
                    }],
                );
            }
            // else: neutral (score between dampen and boost thresholds), no adjustment
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
    fn test_default_config_matches_canonical() {
        let config = ConsciousnessConfig::default();
        let canonical = consciousness_thresholds();
        assert_eq!(config.dampen_threshold, canonical.fl_dampen);
        assert_eq!(config.boost_threshold, canonical.fl_boost);
        assert_eq!(config.dampen_factor, canonical.fl_dampen_factor);
        assert_eq!(config.boost_factor, canonical.fl_boost_factor);
        assert_eq!(config.veto_threshold, canonical.fl_veto);
        assert_eq!(config.default_score, 0.5);
    }

    #[test]
    fn test_custom_config() {
        let config = ConsciousnessConfig {
            dampen_threshold: 0.4,
            boost_threshold: 0.8,
            dampen_factor: 0.1,
            boost_factor: 2.0,
            veto_threshold: 0.05,
            default_score: 0.3,
        };
        let plugin = ConsciousnessAwareByzantinePlugin::with_config(config.clone());
        assert_eq!(plugin.config().dampen_threshold, 0.4);
        assert_eq!(plugin.config().boost_factor, 2.0);
    }

    #[test]
    fn test_high_score_gets_boosted() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        let mut scores = HashMap::new();
        scores.insert("node_a".to_string(), 0.8);
        plugin.set_consciousness_scores(scores);

        let updates = vec![make_update("node_a")];
        let weights = plugin.analyze(&updates);

        assert!(weights.contains_key("node_a"));
        let adj = &weights["node_a"][0];
        assert_eq!(adj.weight_multiplier, 1.5);
        assert!(!adj.veto);
    }

    #[test]
    fn test_low_score_gets_dampened() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        let mut scores = HashMap::new();
        scores.insert("node_b".to_string(), 0.2);
        plugin.set_consciousness_scores(scores);

        let updates = vec![make_update("node_b")];
        let weights = plugin.analyze(&updates);

        assert!(weights.contains_key("node_b"));
        let adj = &weights["node_b"][0];
        assert_eq!(adj.weight_multiplier, 0.3);
        assert!(!adj.veto);
    }

    #[test]
    fn test_very_low_score_gets_vetoed() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        let mut scores = HashMap::new();
        scores.insert("node_c".to_string(), 0.05);
        plugin.set_consciousness_scores(scores);

        let updates = vec![make_update("node_c")];
        let weights = plugin.analyze(&updates);

        assert!(weights.contains_key("node_c"));
        let adj = &weights["node_c"][0];
        assert_eq!(adj.weight_multiplier, 0.0);
        assert!(adj.veto);
    }

    #[test]
    fn test_neutral_score_no_adjustment() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        let mut scores = HashMap::new();
        scores.insert("node_d".to_string(), 0.45); // Between 0.3 and 0.6
        plugin.set_consciousness_scores(scores);

        let updates = vec![make_update("node_d")];
        let weights = plugin.analyze(&updates);

        // Neutral range: no entry in weight map
        assert!(
            !weights.contains_key("node_d"),
            "Neutral score should produce no adjustment"
        );
    }

    #[test]
    fn test_missing_participant_uses_default() {
        let plugin = ConsciousnessAwareByzantinePlugin::new();
        // default_score = 0.5, which is in the neutral range (0.3..0.6)
        assert_eq!(plugin.consciousness_score_for("unknown_node"), 0.5);
    }

    #[test]
    fn test_missing_participant_default_is_neutral() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        // No consciousness scores set — all participants use default (0.5 = neutral)
        let updates = vec![make_update("node_x"), make_update("node_y")];
        let weights = plugin.analyze(&updates);

        assert!(
            weights.is_empty(),
            "Default score (0.5) is neutral, should produce no adjustments"
        );
    }

    #[test]
    fn test_mixed_consciousness_scores() {
        let mut plugin = ConsciousnessAwareByzantinePlugin::new();
        let mut scores = HashMap::new();
        scores.insert("high".to_string(), 0.9);
        scores.insert("mid".to_string(), 0.5);
        scores.insert("low".to_string(), 0.2);
        scores.insert("veto".to_string(), 0.05);
        plugin.set_consciousness_scores(scores);

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
        scores.insert("at_dampen".to_string(), 0.3); // == dampen_threshold → neutral
        scores.insert("at_boost".to_string(), 0.6); // == boost_threshold → neutral
        plugin.set_consciousness_scores(scores);

        let updates = vec![
            make_update("at_veto"),
            make_update("at_dampen"),
            make_update("at_boost"),
        ];
        let weights = plugin.analyze(&updates);

        // at_veto (0.1): >= veto_threshold(0.1), < dampen_threshold(0.3) → dampened
        assert!(weights.contains_key("at_veto"));
        assert!(!weights["at_veto"][0].veto);
        assert_eq!(weights["at_veto"][0].weight_multiplier, 0.3);

        // at_dampen (0.3): >= dampen_threshold(0.3), <= boost_threshold(0.6) → neutral
        assert!(!weights.contains_key("at_dampen"));

        // at_boost (0.6): == boost_threshold(0.6), not > → neutral
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
