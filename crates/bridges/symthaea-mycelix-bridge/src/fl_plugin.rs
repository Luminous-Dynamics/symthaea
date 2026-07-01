// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FL Pipeline Plugin: Consciousness-guided quality assessment for federated learning.
//!
//! This module bridges `SymthaeaBackend` quality assessments into the
//! `mycelix-fl-core` pipeline via the [`ByzantinePlugin`] trait. It provides
//! richer weight adjustments than the simple Phi-threshold plugin by using
//! the full [`QualityScore`] signal: epistemic confidence, anomaly severity,
//! connectivity gain trends, and gray-zone ambiguity detection.
//!
//! # Usage
//!
//! ```ignore
//! use symthaea_mycelix_bridge::fl_plugin::SymthaeaQualityPlugin;
//! use mycelix_fl_core::plugins::PipelinePlugins;
//!
//! // Pre-computed quality scores from SymthaeaBackend::assess_update()
//! let mut plugin = SymthaeaQualityPlugin::new();
//! plugin.set_quality_scores(scores);
//!
//! let mut plugins = PipelinePlugins {
//!     byzantine: vec![&mut plugin],
//!     ..PipelinePlugins::none()
//! };
//! ```

use std::collections::HashMap;

use mycelix_fl_core::pipeline::{ExternalWeightMap, ParticipantWeightAdjustment};
use mycelix_fl_core::plugins::ByzantinePlugin;
use mycelix_fl_core::types::GradientUpdate;

use crate::{ConsciousAnomalySeverity, QualityScore};

/// Configuration for the Symthaea quality plugin.
#[derive(Debug, Clone)]
pub struct QualityPluginConfig {
    /// Minimum epistemic confidence to avoid dampening (default: 0.3).
    pub confidence_threshold: f32,
    /// Epistemic confidence below which the update is vetoed (default: 0.1).
    pub veto_confidence: f32,
    /// Connectivity gain above which the update gets a boost (default: 0.05).
    pub connectivity_gain_boost_threshold: f32,
    /// Weight multiplier for boosted updates (default: 1.4).
    pub boost_factor: f32,
    /// Whether to veto on Severe anomaly regardless of confidence (default: true).
    pub veto_severe: bool,
}

impl Default for QualityPluginConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.3,
            veto_confidence: 0.1,
            connectivity_gain_boost_threshold: 0.05,
            boost_factor: 1.4,
            veto_severe: true,
        }
    }
}

/// Symthaea-backed Byzantine plugin using full quality assessment.
///
/// Unlike `ConsciousnessAwareByzantinePlugin` which uses only raw Phi thresholds,
/// this plugin uses the full `QualityScore` from `SymthaeaBackend::assess_update()`,
/// incorporating:
///
/// - **Epistemic confidence**: composite of Phi + validation accuracy
/// - **Anomaly severity**: None/Mild/Moderate/Severe classification
/// - **Connectivity gain trend**: positive gain -> connectivity improving -> boost
/// - **Gray-zone detection**: ambiguous similarity to known prototypes
///
/// Quality scores must be set externally each round via [`set_quality_scores`].
pub struct SymthaeaQualityPlugin {
    config: QualityPluginConfig,
    /// Pre-computed quality scores per participant, set before each round.
    quality_scores: HashMap<String, QualityScore>,
    /// Rolling count of vetoed participants (for meta-learning feedback).
    total_vetoed: u64,
    /// Rolling count of boosted participants.
    total_boosted: u64,
    /// Rolling count of dampened participants.
    total_dampened: u64,
    /// Number of rounds processed.
    rounds: u64,
}

impl SymthaeaQualityPlugin {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self::with_config(QualityPluginConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: QualityPluginConfig) -> Self {
        Self {
            config,
            quality_scores: HashMap::new(),
            total_vetoed: 0,
            total_boosted: 0,
            total_dampened: 0,
            rounds: 0,
        }
    }

    /// Set quality scores for all participants in the current round.
    ///
    /// Typically produced by running `SymthaeaBackend::assess_update()` on
    /// each participant's gradient before calling the pipeline.
    pub fn set_quality_scores(&mut self, scores: HashMap<String, QualityScore>) {
        self.quality_scores = scores;
    }

    /// Set a single participant's quality score.
    pub fn set_score(&mut self, participant_id: String, score: QualityScore) {
        self.quality_scores.insert(participant_id, score);
    }

    /// Get the quality score for a participant (if available).
    pub fn score_for(&self, participant_id: &str) -> Option<&QualityScore> {
        self.quality_scores.get(participant_id)
    }

    /// Get rolling statistics: (vetoed, boosted, dampened, rounds).
    pub fn stats(&self) -> (u64, u64, u64, u64) {
        (
            self.total_vetoed,
            self.total_boosted,
            self.total_dampened,
            self.rounds,
        )
    }

    /// Access the current configuration.
    pub fn config(&self) -> &QualityPluginConfig {
        &self.config
    }

    /// Compute weight adjustment for a single quality score.
    fn adjustment_for(&self, q: &QualityScore) -> Option<ParticipantWeightAdjustment> {
        // Rule 1: Severe anomaly -> veto (if configured)
        if self.config.veto_severe && matches!(q.severity, ConsciousAnomalySeverity::Severe) {
            return Some(ParticipantWeightAdjustment {
                weight_multiplier: 0.0,
                veto: true,
                source: "symthaea_quality_severe".into(),
            });
        }

        // Rule 2: Very low epistemic confidence -> veto
        if q.epistemic_confidence < self.config.veto_confidence {
            return Some(ParticipantWeightAdjustment {
                weight_multiplier: 0.0,
                veto: true,
                source: "symthaea_quality_low_confidence".into(),
            });
        }

        // Rule 3: Spectral-phi divergence — network looks connected but lacks integration.
        // When spectral connectivity is high but true/fast phi is very low,
        // the network structure doesn't produce genuine information integration.
        let cv = &q.consciousness_vector;
        let spectral = cv.spectral_connectivity.unwrap_or(0.0);
        let best_phi = cv.best_phi();
        if spectral > 0.5 && best_phi < 0.1 && cv.true_phi.or(cv.phi_fast).is_some() {
            return Some(ParticipantWeightAdjustment {
                weight_multiplier: 0.5,
                veto: false,
                source: "symthaea_quality_spectral_phi_divergence".into(),
            });
        }

        // Rule 4: Low confidence or moderate anomaly -> dampen proportionally
        if q.epistemic_confidence < self.config.confidence_threshold
            || matches!(q.severity, ConsciousAnomalySeverity::Moderate)
        {
            let multiplier = q.epistemic_confidence.clamp(0.1, 1.0);
            return Some(ParticipantWeightAdjustment {
                weight_multiplier: multiplier,
                veto: false,
                source: "symthaea_quality_dampen".into(),
            });
        }

        // Rule 5: Mild anomaly with ambiguity -> slight dampen
        if q.is_ambiguous && matches!(q.severity, ConsciousAnomalySeverity::Mild) {
            return Some(ParticipantWeightAdjustment {
                weight_multiplier: 0.8,
                veto: false,
                source: "symthaea_quality_ambiguous".into(),
            });
        }

        // Rule 6: High confidence + positive connectivity gain -> boost
        // Uses best_phi from C-Vector for richer signal
        if q.epistemic_confidence > 0.7
            && q.spectral.connectivity_gain > self.config.connectivity_gain_boost_threshold
        {
            return Some(ParticipantWeightAdjustment {
                weight_multiplier: self.config.boost_factor,
                veto: false,
                source: "symthaea_quality_boost".into(),
            });
        }

        // No adjustment for neutral updates
        None
    }
}

impl Default for SymthaeaQualityPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl ByzantinePlugin for SymthaeaQualityPlugin {
    fn analyze(&mut self, updates: &[GradientUpdate]) -> ExternalWeightMap {
        let mut weights = ExternalWeightMap::new();

        for update in updates {
            if let Some(quality) = self.quality_scores.get(&update.participant_id) {
                if let Some(adj) = self.adjustment_for(quality) {
                    if adj.veto {
                        self.total_vetoed += 1;
                    } else if adj.weight_multiplier > 1.0 {
                        self.total_boosted += 1;
                    } else if adj.weight_multiplier < 1.0 {
                        self.total_dampened += 1;
                    }
                    weights.insert(update.participant_id.clone(), vec![adj]);
                }
            }
        }

        weights
    }

    fn name(&self) -> &str {
        "symthaea_quality"
    }

    fn record_outcome(&mut self, _round: u64, _excluded_ids: &[String]) {
        self.rounds += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConsciousnessVector, SpectralConnectivityAssessment};

    fn make_update(id: &str) -> GradientUpdate {
        GradientUpdate::new(id.into(), 1, vec![0.5; 10], 100, 0.5)
    }

    fn good_quality() -> QualityScore {
        QualityScore {
            accuracy: 0.9,
            loss: 0.1,
            spectral: SpectralConnectivityAssessment::new(0.4, 0.6),
            consciousness_vector: ConsciousnessVector::default(),
            epistemic_confidence: 0.85,
            is_anomalous: false,
            similarity: Some(0.95),
            is_ambiguous: false,
            severity: ConsciousAnomalySeverity::None,
            causes: Vec::new(),
        }
    }

    fn bad_quality() -> QualityScore {
        QualityScore {
            accuracy: 0.2,
            loss: 2.0,
            spectral: SpectralConnectivityAssessment::new(0.6, 0.1),
            consciousness_vector: ConsciousnessVector::default(),
            epistemic_confidence: 0.05,
            is_anomalous: true,
            similarity: Some(0.3),
            is_ambiguous: false,
            severity: ConsciousAnomalySeverity::Severe,
            causes: vec!["phi_drop".into(), "low_confidence".into()],
        }
    }

    fn moderate_quality() -> QualityScore {
        QualityScore {
            accuracy: 0.5,
            loss: 0.8,
            spectral: SpectralConnectivityAssessment::new(0.5, 0.35),
            consciousness_vector: ConsciousnessVector::default(),
            epistemic_confidence: 0.25,
            is_anomalous: true,
            similarity: Some(0.6),
            is_ambiguous: false,
            severity: ConsciousAnomalySeverity::Moderate,
            causes: vec!["phi_drop".into()],
        }
    }

    fn ambiguous_quality() -> QualityScore {
        QualityScore {
            accuracy: 0.6,
            loss: 0.5,
            spectral: SpectralConnectivityAssessment::new(0.5, 0.52),
            consciousness_vector: ConsciousnessVector::default(),
            epistemic_confidence: 0.55,
            is_anomalous: true,
            similarity: Some(0.8),
            is_ambiguous: true,
            severity: ConsciousAnomalySeverity::Mild,
            causes: vec!["gray_zone".into()],
        }
    }

    #[test]
    fn test_severe_anomaly_vetoed() {
        let mut plugin = SymthaeaQualityPlugin::new();
        let mut scores = HashMap::new();
        scores.insert("bad_node".to_string(), bad_quality());
        plugin.set_quality_scores(scores);

        let updates = vec![make_update("bad_node")];
        let weights = plugin.analyze(&updates);

        assert!(weights.contains_key("bad_node"));
        assert!(weights["bad_node"][0].veto);
        assert_eq!(weights["bad_node"][0].weight_multiplier, 0.0);
    }

    #[test]
    fn test_low_confidence_vetoed() {
        let mut plugin = SymthaeaQualityPlugin::new();
        let mut q = good_quality();
        q.epistemic_confidence = 0.05;
        q.severity = ConsciousAnomalySeverity::None;

        let mut scores = HashMap::new();
        scores.insert("low_conf".to_string(), q);
        plugin.set_quality_scores(scores);

        let updates = vec![make_update("low_conf")];
        let weights = plugin.analyze(&updates);

        assert!(weights.contains_key("low_conf"));
        assert!(weights["low_conf"][0].veto);
    }

    #[test]
    fn test_moderate_anomaly_dampened() {
        let mut plugin = SymthaeaQualityPlugin::new();
        let mut scores = HashMap::new();
        scores.insert("moderate".to_string(), moderate_quality());
        plugin.set_quality_scores(scores);

        let updates = vec![make_update("moderate")];
        let weights = plugin.analyze(&updates);

        assert!(weights.contains_key("moderate"));
        let adj = &weights["moderate"][0];
        assert!(!adj.veto);
        assert!(adj.weight_multiplier < 1.0, "Should be dampened");
        assert!(adj.weight_multiplier > 0.0, "Should not be vetoed");
    }

    #[test]
    fn test_ambiguous_mild_dampened() {
        let mut plugin = SymthaeaQualityPlugin::new();
        let mut scores = HashMap::new();
        scores.insert("ambig".to_string(), ambiguous_quality());
        plugin.set_quality_scores(scores);

        let updates = vec![make_update("ambig")];
        let weights = plugin.analyze(&updates);

        assert!(weights.contains_key("ambig"));
        let adj = &weights["ambig"][0];
        assert!(!adj.veto);
        assert_eq!(adj.weight_multiplier, 0.8);
        assert_eq!(adj.source, "symthaea_quality_ambiguous");
    }

    #[test]
    fn test_good_quality_boosted() {
        let mut plugin = SymthaeaQualityPlugin::new();
        let mut q = good_quality();
        q.spectral.connectivity_gain = 0.2;
        q.epistemic_confidence = 0.85;

        let mut scores = HashMap::new();
        scores.insert("good".to_string(), q);
        plugin.set_quality_scores(scores);

        let updates = vec![make_update("good")];
        let weights = plugin.analyze(&updates);

        assert!(weights.contains_key("good"));
        let adj = &weights["good"][0];
        assert!(!adj.veto);
        assert_eq!(adj.weight_multiplier, 1.4);
        assert_eq!(adj.source, "symthaea_quality_boost");
    }

    #[test]
    fn test_neutral_no_adjustment() {
        let mut plugin = SymthaeaQualityPlugin::new();
        let mut q = good_quality();
        q.epistemic_confidence = 0.5;
        q.spectral.connectivity_gain = 0.01;

        let mut scores = HashMap::new();
        scores.insert("neutral".to_string(), q);
        plugin.set_quality_scores(scores);

        let updates = vec![make_update("neutral")];
        let weights = plugin.analyze(&updates);

        assert!(
            !weights.contains_key("neutral"),
            "Neutral should have no adjustment"
        );
    }

    #[test]
    fn test_missing_score_no_adjustment() {
        let mut plugin = SymthaeaQualityPlugin::new();
        let updates = vec![make_update("unknown")];
        let weights = plugin.analyze(&updates);

        assert!(weights.is_empty());
    }

    #[test]
    fn test_mixed_participants() {
        let mut plugin = SymthaeaQualityPlugin::new();
        let mut scores = HashMap::new();

        let mut boost = good_quality();
        boost.spectral.connectivity_gain = 0.2;
        boost.epistemic_confidence = 0.85;
        scores.insert("boost".to_string(), boost);

        scores.insert("veto".to_string(), bad_quality());
        scores.insert("moderate".to_string(), moderate_quality());

        plugin.set_quality_scores(scores);

        let updates = vec![
            make_update("boost"),
            make_update("veto"),
            make_update("moderate"),
        ];
        let weights = plugin.analyze(&updates);

        assert!(weights["boost"][0].weight_multiplier > 1.0);
        assert!(!weights["boost"][0].veto);
        assert!(weights["veto"][0].veto);
        assert!(!weights["moderate"][0].veto);
        assert!(weights["moderate"][0].weight_multiplier < 1.0);
    }

    #[test]
    fn test_stats_tracking() {
        let mut plugin = SymthaeaQualityPlugin::new();
        let mut scores = HashMap::new();

        let mut boost_q = good_quality();
        boost_q.spectral.connectivity_gain = 0.2;
        boost_q.epistemic_confidence = 0.85;
        scores.insert("boost".to_string(), boost_q);
        scores.insert("veto".to_string(), bad_quality());
        scores.insert("dampen".to_string(), moderate_quality());

        plugin.set_quality_scores(scores);

        let updates = vec![
            make_update("boost"),
            make_update("veto"),
            make_update("dampen"),
        ];
        let _ = plugin.analyze(&updates);

        let (vetoed, boosted, dampened, rounds) = plugin.stats();
        assert_eq!(vetoed, 1);
        assert_eq!(boosted, 1);
        assert_eq!(dampened, 1);
        assert_eq!(rounds, 0);

        plugin.record_outcome(1, &["veto".to_string()]);
        let (_, _, _, rounds) = plugin.stats();
        assert_eq!(rounds, 1);
    }

    #[test]
    fn test_custom_config() {
        let config = QualityPluginConfig {
            confidence_threshold: 0.5,
            veto_confidence: 0.2,
            connectivity_gain_boost_threshold: 0.1,
            boost_factor: 2.0,
            veto_severe: false,
        };
        let mut plugin = SymthaeaQualityPlugin::with_config(config);

        let mut scores = HashMap::new();
        let mut severe = bad_quality();
        severe.epistemic_confidence = 0.25;
        scores.insert("severe".to_string(), severe);
        plugin.set_quality_scores(scores);

        let updates = vec![make_update("severe")];
        let weights = plugin.analyze(&updates);

        assert!(weights.contains_key("severe"));
        assert!(!weights["severe"][0].veto);
        assert!(weights["severe"][0].weight_multiplier < 1.0);
    }

    #[test]
    fn test_plugin_name() {
        let plugin = SymthaeaQualityPlugin::new();
        assert_eq!(ByzantinePlugin::name(&plugin), "symthaea_quality");
    }

    #[test]
    fn test_set_individual_score() {
        let mut plugin = SymthaeaQualityPlugin::new();
        plugin.set_score("node_a".to_string(), good_quality());

        assert!(plugin.score_for("node_a").is_some());
        assert!(plugin.score_for("node_b").is_none());
    }

    #[test]
    fn test_empty_updates() {
        let mut plugin = SymthaeaQualityPlugin::new();
        let weights = plugin.analyze(&[]);
        assert!(weights.is_empty());
    }

    #[test]
    fn test_spectral_phi_divergence_detected() {
        let mut plugin = SymthaeaQualityPlugin::new();

        // High spectral connectivity but very low phi -> divergence signal
        let mut q = good_quality();
        q.consciousness_vector = ConsciousnessVector {
            spectral_connectivity: Some(0.8),
            true_phi: Some(0.05), // Very low true phi
            phi_fast: Some(0.03),
            entropy: Some(0.5),
            coherence: Some(0.7),
            epistemic_confidence: Some(0.6),
        };
        q.epistemic_confidence = 0.6;

        let mut scores = HashMap::new();
        scores.insert("divergent".to_string(), q);
        plugin.set_quality_scores(scores);

        let updates = vec![make_update("divergent")];
        let weights = plugin.analyze(&updates);

        assert!(weights.contains_key("divergent"));
        let adj = &weights["divergent"][0];
        assert!(!adj.veto, "divergence should dampen, not veto");
        assert_eq!(adj.weight_multiplier, 0.5);
        assert_eq!(adj.source, "symthaea_quality_spectral_phi_divergence");
    }

    #[test]
    fn test_no_divergence_when_phi_not_computed() {
        let mut plugin = SymthaeaQualityPlugin::new();

        // High spectral but no phi computed -> no divergence detection
        let mut q = good_quality();
        q.consciousness_vector = ConsciousnessVector {
            spectral_connectivity: Some(0.8),
            true_phi: None,
            phi_fast: None,
            entropy: Some(0.5),
            coherence: Some(0.7),
            epistemic_confidence: Some(0.6),
        };
        q.epistemic_confidence = 0.6;
        q.spectral.connectivity_gain = 0.01; // below boost threshold

        let mut scores = HashMap::new();
        scores.insert("no_phi".to_string(), q);
        plugin.set_quality_scores(scores);

        let updates = vec![make_update("no_phi")];
        let weights = plugin.analyze(&updates);

        // Should have no adjustment (not divergence, not dampened, not boosted)
        assert!(
            !weights.contains_key("no_phi"),
            "No divergence without phi data"
        );
    }
}
