//! Trust management for federated learning participants.
//!
//! Provides reputation tracking, trust score computation, and K-Vector
//! trust profiles for each participant in the FL network.
//!
//! # Status
//! Stub implementation — trust score computation and reputation decay are
//! functional. Full K-Vector trust integration is scaffolded.

use serde::{Deserialize, Serialize};

/// A participant's trust profile, updated each round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustProfile {
    /// Participant identifier
    pub participant_id: String,
    /// Composite trust score in [0.0, 1.0]
    pub trust_score: f32,
    /// Number of successful rounds contributed
    pub successful_rounds: u32,
    /// Number of rounds flagged as Byzantine
    pub byzantine_rounds: u32,
    /// Exponential moving average of gradient quality
    pub quality_ema: f32,
    /// Unix timestamp of last update
    pub last_updated: i64,
}

impl TrustProfile {
    /// Create a new participant with neutral trust.
    pub fn new(participant_id: String, now: i64) -> Self {
        Self {
            participant_id,
            trust_score: 0.5,
            successful_rounds: 0,
            byzantine_rounds: 0,
            quality_ema: 0.5,
            last_updated: now,
        }
    }

    /// Update trust after a successful round.
    pub fn reward(&mut self, quality_score: f32, now: i64) {
        self.successful_rounds += 1;
        // EMA with alpha=0.1
        self.quality_ema = 0.9 * self.quality_ema + 0.1 * quality_score;
        self.trust_score = self.compute_composite();
        self.last_updated = now;
    }

    /// Penalize trust after Byzantine detection.
    pub fn penalize(&mut self, confidence: f32, now: i64) {
        self.byzantine_rounds += 1;
        // Penalize proportional to detection confidence
        self.trust_score = (self.trust_score - 0.3 * confidence).max(0.0);
        self.last_updated = now;
    }

    /// Compute composite trust from success/failure ratio and quality EMA.
    fn compute_composite(&self) -> f32 {
        let total = self.successful_rounds + self.byzantine_rounds;
        if total == 0 {
            return 0.5;
        }
        let success_rate = self.successful_rounds as f32 / total as f32;
        // Weighted average: 60% success rate, 40% quality EMA
        (0.6 * success_rate + 0.4 * self.quality_ema).clamp(0.0, 1.0)
    }

    /// Apply exponential time-based decay.
    ///
    /// Trust decays toward a floor of 0.1 when no activity occurs.
    pub fn apply_decay(&mut self, elapsed_seconds: i64, now: i64) {
        const DECAY_INTERVAL: i64 = 86400; // 1 day
        const DECAY_FACTOR: f64 = 0.95;
        const FLOOR: f32 = 0.1;

        if elapsed_seconds <= 0 {
            return;
        }
        let intervals = elapsed_seconds as f64 / DECAY_INTERVAL as f64;
        let decayed = FLOOR as f64
            + (self.trust_score as f64 - FLOOR as f64) * DECAY_FACTOR.powf(intervals);
        self.trust_score = decayed.clamp(FLOOR as f64, 1.0) as f32;
        self.last_updated = now;
    }
}

/// In-memory trust registry for the current session.
///
/// In production, trust profiles are stored on the DHT.
/// This struct provides a local cache for pipeline computations.
#[derive(Debug, Default)]
pub struct TrustRegistry {
    profiles: std::collections::HashMap<String, TrustProfile>,
}

impl TrustRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a trust profile for a participant.
    pub fn get_or_default(&mut self, participant_id: &str, now: i64) -> &mut TrustProfile {
        self.profiles
            .entry(participant_id.to_string())
            .or_insert_with(|| TrustProfile::new(participant_id.to_string(), now))
    }

    /// Get the trust score for a participant (returns 0.5 if unknown).
    pub fn trust_score(&self, participant_id: &str) -> f32 {
        self.profiles
            .get(participant_id)
            .map(|p| p.trust_score)
            .unwrap_or(0.5)
    }

    /// Build a reputation map for use with the pipeline.
    pub fn to_reputation_map(&self) -> std::collections::HashMap<String, f32> {
        self.profiles
            .iter()
            .map(|(id, p)| (id.clone(), p.trust_score))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_profile_neutral_trust() {
        let p = TrustProfile::new("p1".to_string(), 0);
        assert!((p.trust_score - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_reward_increases_trust() {
        let mut p = TrustProfile::new("p1".to_string(), 0);
        p.reward(1.0, 100);
        assert!(p.trust_score >= 0.5);
    }

    #[test]
    fn test_penalize_decreases_trust() {
        let mut p = TrustProfile::new("p1".to_string(), 0);
        p.penalize(0.9, 100);
        assert!(p.trust_score < 0.5);
    }

    #[test]
    fn test_decay_reduces_trust() {
        let mut p = TrustProfile::new("p1".to_string(), 0);
        p.trust_score = 0.8;
        p.apply_decay(86400 * 7, 86400 * 7); // 7 days
        assert!(p.trust_score < 0.8);
        assert!(p.trust_score >= 0.1);
    }

    #[test]
    fn test_registry_default_score() {
        let registry = TrustRegistry::new();
        assert!((registry.trust_score("unknown") - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_registry_reputation_map() {
        let mut registry = TrustRegistry::new();
        registry.get_or_default("p1", 0).trust_score = 0.9;
        let map = registry.to_reputation_map();
        assert!((map["p1"] - 0.9).abs() < 0.001);
    }
}
