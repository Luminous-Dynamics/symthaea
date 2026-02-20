//! Meta-Learning Byzantine Plugin
//!
//! Learns from past aggregation outcomes to improve Byzantine detection.
//! Tracks per-participant exclusion history (EMA) and adapts signal weights
//! based on which detection signals correctly predicted exclusions.
//!
//! # Novel Contribution
//!
//! No existing FL framework implements meta-learning for Byzantine detection.
//! This plugin learns:
//! 1. Which participants are persistently Byzantine (EMA exclusion rate)
//! 2. Which detection signals are most predictive (signal weight adaptation)

use std::collections::HashMap;

use crate::byzantine::{MultiSignalByzantineDetector, SignalWeights};
use crate::pipeline::ExternalWeightMap;
use crate::plugins::ByzantinePlugin;
use crate::types::GradientUpdate;

/// Configuration for the meta-learning Byzantine plugin.
///
/// **Threshold disambiguation**: The thresholds here (`ema_alpha`, `suspicion_threshold`,
/// `learning_rate`) are meta-learning signal parameters — they are NOT IIT Phi thresholds
/// and NOT governance consciousness gates. For canonical consciousness thresholds used in
/// governance and consciousness-aware FL, see `mycelix_bridge_common::consciousness_thresholds`.
#[derive(Debug, Clone)]
pub struct MetaLearningConfig {
    /// EMA smoothing factor for exclusion rate (0.0-1.0).
    /// Higher = faster adaptation, lower = more stable.
    pub ema_alpha: f32,
    /// Exclusion rate above which a participant is flagged as suspicious.
    /// This is a meta-learning signal threshold, not an IIT Phi threshold.
    pub suspicion_threshold: f32,
    /// Learning rate for signal weight adaptation.
    pub learning_rate: f32,
    /// Minimum rounds of history before trusting exclusion rates.
    pub min_rounds: u32,
    /// Weight multiplier applied to suspicious participants (0.0-1.0).
    /// Lower = more aggressive dampening.
    pub suspicious_weight: f32,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            ema_alpha: 0.1,
            suspicion_threshold: 0.3,
            learning_rate: 0.01,
            min_rounds: 5,
            suspicious_weight: 0.2,
        }
    }
}

/// Per-participant profile tracked by the meta-learning plugin.
#[derive(Debug, Clone)]
pub struct ParticipantProfile {
    /// Exponential moving average of exclusion rate (0.0-1.0).
    pub exclusion_rate: f32,
    /// Total rounds this participant has been seen.
    pub rounds_seen: u32,
    /// Total times this participant was excluded.
    pub exclusion_count: u32,
    /// Last round's anomaly scores per signal.
    pub last_signal_scores: Option<[f32; 4]>,
}

impl ParticipantProfile {
    fn new() -> Self {
        Self {
            exclusion_rate: 0.0,
            rounds_seen: 0,
            exclusion_count: 0,
            last_signal_scores: None,
        }
    }
}

/// Meta-learning Byzantine detection plugin.
///
/// Implements `ByzantinePlugin` to plug into the unified pipeline.
/// Maintains per-participant exclusion history and adapts detection
/// signal weights based on feedback from past rounds.
pub struct MetaLearningByzantinePlugin {
    /// Per-participant exclusion history (EMA).
    profiles: HashMap<String, ParticipantProfile>,
    /// Learned signal weights.
    signal_weights: SignalWeights,
    /// Configuration.
    config: MetaLearningConfig,
    /// Total rounds processed.
    rounds_processed: u64,
    /// Cached signal scores from last analyze() call, keyed by participant_id.
    last_analysis_scores: HashMap<String, [f32; 4]>,
}

impl MetaLearningByzantinePlugin {
    /// Create a new meta-learning plugin with default configuration.
    pub fn new() -> Self {
        Self::with_config(MetaLearningConfig::default())
    }

    /// Create a new meta-learning plugin with custom configuration.
    pub fn with_config(config: MetaLearningConfig) -> Self {
        Self {
            profiles: HashMap::new(),
            signal_weights: SignalWeights::default(),
            config,
            rounds_processed: 0,
            last_analysis_scores: HashMap::new(),
        }
    }

    /// Get the profile for a specific participant.
    pub fn get_participant_profile(&self, id: &str) -> Option<&ParticipantProfile> {
        self.profiles.get(id)
    }

    /// Get all participant profiles.
    pub fn all_profiles(&self) -> &HashMap<String, ParticipantProfile> {
        &self.profiles
    }

    /// Get the current learned signal weights.
    pub fn signal_weights(&self) -> &SignalWeights {
        &self.signal_weights
    }

    /// Get total rounds processed.
    pub fn rounds_processed(&self) -> u64 {
        self.rounds_processed
    }

    /// Check if a participant is considered suspicious based on history.
    pub fn is_suspicious(&self, id: &str) -> bool {
        if let Some(profile) = self.profiles.get(id) {
            profile.rounds_seen >= self.config.min_rounds
                && profile.exclusion_rate > self.config.suspicion_threshold
        } else {
            false
        }
    }
}

impl Default for MetaLearningByzantinePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl ByzantinePlugin for MetaLearningByzantinePlugin {
    fn analyze(&mut self, updates: &[GradientUpdate]) -> ExternalWeightMap {
        let mut weights = ExternalWeightMap::new();
        self.last_analysis_scores.clear();

        if updates.len() < 3 {
            return weights;
        }

        // Run multi-signal detection with current learned weights
        let detector = MultiSignalByzantineDetector::new();
        let detection = detector.detect(updates);

        // Store per-participant signal scores for later feedback
        for breakdown in &detection.signal_breakdown {
            if let Some(update) = updates.get(breakdown.participant_idx) {
                let scores = [
                    breakdown.magnitude_score,
                    breakdown.direction_score,
                    breakdown.cross_validation_score,
                    breakdown.coordinate_score,
                ];
                self.last_analysis_scores
                    .insert(update.participant_id.clone(), scores);

                // Update profile with latest scores
                let profile = self
                    .profiles
                    .entry(update.participant_id.clone())
                    .or_insert_with(ParticipantProfile::new);
                profile.last_signal_scores = Some(scores);
            }
        }

        // Compute weighted anomaly score using learned weights
        for update in updates {
            let scores = match self.last_analysis_scores.get(&update.participant_id) {
                Some(s) => s,
                None => continue,
            };

            let weighted_score = self.signal_weights.magnitude * scores[0]
                + self.signal_weights.direction * scores[1]
                + self.signal_weights.cross_validation * scores[2]
                + self.signal_weights.coordinate * scores[3];

            let profile = self.profiles.get(&update.participant_id);
            let history_suspicious = profile
                .map(|p| {
                    p.rounds_seen >= self.config.min_rounds
                        && p.exclusion_rate > self.config.suspicion_threshold
                })
                .unwrap_or(false);

            // Flag if weighted anomaly score is high OR historical exclusion rate is suspicious
            if weighted_score > 0.5 || history_suspicious {
                let multiplier = if history_suspicious {
                    self.config.suspicious_weight
                } else {
                    (1.0 - weighted_score).max(0.1)
                };

                weights.insert(
                    update.participant_id.clone(),
                    vec![crate::pipeline::ParticipantWeightAdjustment {
                        weight_multiplier: multiplier,
                        veto: weighted_score > 0.8 && history_suspicious,
                        source: "meta_learning_byzantine".into(),
                    }],
                );
            }
        }

        weights
    }

    fn name(&self) -> &str {
        "meta_learning_byzantine"
    }

    fn record_outcome(&mut self, _round: u64, excluded_ids: &[String]) {
        self.rounds_processed += 1;
        let excluded_set: std::collections::HashSet<&String> = excluded_ids.iter().collect();

        // Update EMA exclusion rates for all known participants
        for (pid, profile) in self.profiles.iter_mut() {
            let was_excluded = excluded_set.contains(pid);
            let observation = if was_excluded { 1.0_f32 } else { 0.0 };

            // EMA update: rate = alpha * observation + (1 - alpha) * rate
            profile.exclusion_rate = self.config.ema_alpha * observation
                + (1.0 - self.config.ema_alpha) * profile.exclusion_rate;
            profile.rounds_seen += 1;
            if was_excluded {
                profile.exclusion_count += 1;
            }
        }

        // Adapt signal weights based on which signals predicted the outcome.
        // For each excluded participant, increase weight of signals that flagged them.
        // For each non-excluded participant, decrease weight of signals that flagged them.
        let mut weight_deltas = [0.0_f32; 4];
        let mut delta_count = 0;

        for (pid, scores) in &self.last_analysis_scores {
            let was_excluded = excluded_set.contains(pid);

            for (i, &score) in scores.iter().enumerate() {
                let predicted_excluded = score > 0.5;
                if predicted_excluded == was_excluded {
                    // Correct prediction: reinforce this signal
                    weight_deltas[i] += self.config.learning_rate;
                } else {
                    // Wrong prediction: weaken this signal
                    weight_deltas[i] -= self.config.learning_rate;
                }
            }
            delta_count += 1;
        }

        if delta_count > 0 {
            // Normalize deltas
            let n = delta_count as f32;
            for delta in &mut weight_deltas {
                *delta /= n;
            }

            // Apply deltas with clamping
            self.signal_weights.magnitude =
                (self.signal_weights.magnitude + weight_deltas[0]).clamp(0.05, 0.6);
            self.signal_weights.direction =
                (self.signal_weights.direction + weight_deltas[1]).clamp(0.05, 0.6);
            self.signal_weights.cross_validation =
                (self.signal_weights.cross_validation + weight_deltas[2]).clamp(0.05, 0.6);
            self.signal_weights.coordinate =
                (self.signal_weights.coordinate + weight_deltas[3]).clamp(0.05, 0.6);

            // Renormalize to sum to 1.0
            let total = self.signal_weights.magnitude
                + self.signal_weights.direction
                + self.signal_weights.cross_validation
                + self.signal_weights.coordinate;
            if total > 0.0 {
                self.signal_weights.magnitude /= total;
                self.signal_weights.direction /= total;
                self.signal_weights.cross_validation /= total;
                self.signal_weights.coordinate /= total;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_updates(n_honest: usize, n_byzantine: usize) -> Vec<GradientUpdate> {
        let mut updates = Vec::new();
        for i in 0..n_honest {
            let val = 0.5 + (i as f32 * 0.001);
            updates.push(GradientUpdate::new(
                format!("honest_{}", i),
                1,
                vec![val; 20],
                100,
                0.5,
            ));
        }
        for i in 0..n_byzantine {
            let val = if i % 2 == 0 { 100.0 } else { -100.0 };
            updates.push(GradientUpdate::new(
                format!("byz_{}", i),
                1,
                vec![val; 20],
                100,
                0.5,
            ));
        }
        updates
    }

    #[test]
    fn test_cold_start_all_trusted() {
        let mut plugin = MetaLearningByzantinePlugin::new();
        let updates = make_updates(10, 0);
        let weights = plugin.analyze(&updates);
        // No Byzantine → no weight adjustments (or very minor ones)
        let vetoed: Vec<_> = weights
            .values()
            .filter(|adjs| adjs.iter().any(|a| a.veto))
            .collect();
        assert!(vetoed.is_empty(), "Cold start should not veto honest nodes");
    }

    #[test]
    fn test_detects_byzantine_on_first_round() {
        let mut plugin = MetaLearningByzantinePlugin::new();
        let updates = make_updates(8, 2);
        let weights = plugin.analyze(&updates);
        // Byzantine nodes should get weight adjustments (not necessarily vetoes on first round)
        let has_adjustments = weights.keys().any(|k| k.starts_with("byz_"));
        assert!(
            has_adjustments,
            "Should flag Byzantine nodes on first round"
        );
    }

    #[test]
    fn test_persistent_attacker_flagged_within_5_rounds() {
        let mut plugin = MetaLearningByzantinePlugin::new();

        // Simulate 7 rounds where byz_0 and byz_1 are always excluded
        for _ in 0..7 {
            let updates = make_updates(8, 2);
            let _weights = plugin.analyze(&updates);
            plugin.record_outcome(
                plugin.rounds_processed,
                &["byz_0".to_string(), "byz_1".to_string()],
            );
        }

        // After 7 rounds, persistent attackers should be flagged as suspicious
        assert!(
            plugin.is_suspicious("byz_0"),
            "byz_0 should be suspicious after 7 exclusions"
        );
        assert!(
            plugin.is_suspicious("byz_1"),
            "byz_1 should be suspicious after 7 exclusions"
        );
    }

    #[test]
    fn test_reformed_participant_decays() {
        let mut plugin = MetaLearningByzantinePlugin::new();

        // 5 rounds of exclusion
        for _ in 0..5 {
            let updates = make_updates(8, 2);
            let _weights = plugin.analyze(&updates);
            plugin.record_outcome(
                plugin.rounds_processed,
                &["byz_0".to_string(), "byz_1".to_string()],
            );
        }

        let rate_after_exclusion = plugin
            .get_participant_profile("byz_0")
            .unwrap()
            .exclusion_rate;

        // 20 rounds of NOT being excluded (reformed)
        for _ in 0..20 {
            let updates = make_updates(10, 0); // all honest
                                               // Ensure byz_0 is present as an honest node
            let mut updates_with_reformed = updates;
            updates_with_reformed.push(GradientUpdate::new(
                "byz_0".to_string(),
                1,
                vec![0.5; 20],
                100,
                0.5,
            ));
            let _weights = plugin.analyze(&updates_with_reformed);
            plugin.record_outcome(plugin.rounds_processed, &[]); // nobody excluded
        }

        let rate_after_reform = plugin
            .get_participant_profile("byz_0")
            .unwrap()
            .exclusion_rate;

        assert!(
            rate_after_reform < rate_after_exclusion,
            "Exclusion rate should decay after reform: {} should be < {}",
            rate_after_reform,
            rate_after_exclusion
        );
    }

    #[test]
    fn test_signal_weight_adaptation() {
        let mut plugin = MetaLearningByzantinePlugin::new();
        let initial_magnitude = plugin.signal_weights().magnitude;

        // Run 10 rounds with magnitude-based attacks (extreme norms)
        for _ in 0..10 {
            let updates = make_updates(8, 2);
            let _weights = plugin.analyze(&updates);
            // Exclude Byzantine nodes (which trigger magnitude signal strongly)
            plugin.record_outcome(
                plugin.rounds_processed,
                &["byz_0".to_string(), "byz_1".to_string()],
            );
        }

        let adapted_magnitude = plugin.signal_weights().magnitude;
        // Signal weights should have changed (adapted based on feedback)
        assert!(
            (adapted_magnitude - initial_magnitude).abs() > 0.001,
            "Signal weights should adapt over 10 rounds: initial={}, adapted={}",
            initial_magnitude,
            adapted_magnitude
        );
    }

    #[test]
    fn test_false_positive_protection() {
        let mut plugin = MetaLearningByzantinePlugin::new();

        // 1 exclusion followed by 10 inclusions
        let updates = make_updates(10, 0);
        let _weights = plugin.analyze(&updates);
        plugin.record_outcome(
            plugin.rounds_processed,
            &["honest_0".to_string()], // falsely excluded once
        );

        for _ in 0..10 {
            let updates = make_updates(10, 0);
            let _weights = plugin.analyze(&updates);
            plugin.record_outcome(plugin.rounds_processed, &[]); // no exclusions
        }

        // After 1 exclusion and 10 non-exclusions, should NOT be suspicious
        assert!(
            !plugin.is_suspicious("honest_0"),
            "One false positive should not make a participant permanently suspicious"
        );
    }

    #[test]
    fn test_exclusion_rate_ema() {
        let mut plugin = MetaLearningByzantinePlugin::new();
        let alpha = plugin.config.ema_alpha;

        let updates = make_updates(5, 0);
        let _weights = plugin.analyze(&updates);

        // Round 1: excluded
        plugin.record_outcome(1, &["honest_0".to_string()]);
        let rate1 = plugin
            .get_participant_profile("honest_0")
            .unwrap()
            .exclusion_rate;
        assert!(
            (rate1 - alpha).abs() < 0.001,
            "After 1 exclusion, rate should be alpha={}, got {}",
            alpha,
            rate1
        );

        // Round 2: not excluded
        let _weights = plugin.analyze(&updates);
        plugin.record_outcome(2, &[]);
        let rate2 = plugin
            .get_participant_profile("honest_0")
            .unwrap()
            .exclusion_rate;
        let expected = alpha * 0.0 + (1.0 - alpha) * rate1;
        assert!(
            (rate2 - expected).abs() < 0.001,
            "After 1 exclusion + 1 inclusion, rate should be {}, got {}",
            expected,
            rate2
        );
    }

    #[test]
    fn test_rounds_counter() {
        let mut plugin = MetaLearningByzantinePlugin::new();
        assert_eq!(plugin.rounds_processed(), 0);

        for i in 0..5 {
            let updates = make_updates(5, 0);
            let _weights = plugin.analyze(&updates);
            plugin.record_outcome(i + 1, &[]);
        }

        assert_eq!(plugin.rounds_processed(), 5);
    }

    #[test]
    fn test_integration_with_pipeline_plugins() {
        // Verify MetaLearningByzantinePlugin implements ByzantinePlugin correctly
        let mut plugin = MetaLearningByzantinePlugin::new();
        assert_eq!(plugin.name(), "meta_learning_byzantine");

        let updates = make_updates(8, 2);
        let weights: ExternalWeightMap = plugin.analyze(&updates);
        // Should return a valid ExternalWeightMap
        assert!(weights.len() <= updates.len());

        plugin.record_outcome(1, &["byz_0".to_string()]);
        assert_eq!(plugin.rounds_processed(), 1);
    }

    #[test]
    fn test_signal_weights_sum_to_one() {
        let mut plugin = MetaLearningByzantinePlugin::new();

        // After multiple rounds of adaptation, weights should still sum to ~1.0
        for _ in 0..20 {
            let updates = make_updates(8, 2);
            let _weights = plugin.analyze(&updates);
            plugin.record_outcome(
                plugin.rounds_processed,
                &["byz_0".to_string(), "byz_1".to_string()],
            );
        }

        let w = plugin.signal_weights();
        let sum = w.magnitude + w.direction + w.cross_validation + w.coordinate;
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Signal weights should sum to ~1.0, got {}",
            sum
        );
    }
}
