//! PoGQ Enhanced (v4.1-Lite)
//!
//! Lightweight implementation of PoGQ-v4.1 Phase 2 enhancements for WASM:
//! - Temporal EMA smoothing (β=0.85)
//! - Warm-up quota (W=3 rounds grace period)
//! - Hysteresis (k=2 violations to quarantine, m=3 to release)
//! - Egregious cap (p99.9 threshold during warm-up)
//!
//! This is a pure-Rust implementation suitable for Holochain zomes.

// PoGQ v4.1 Enhanced types — configuration and evaluation structs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for PoGQ v4.1 Enhanced (Lite)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoGQv41Config {
    /// Weight for historical scores in EMA (0.85 = stable but responsive)
    pub ema_beta: f32,

    /// Number of rounds before full enforcement
    pub warmup_rounds: u32,

    /// Consecutive violations required to quarantine
    pub hysteresis_k: u32,

    /// Consecutive clears required to release from quarantine
    pub hysteresis_m: u32,

    /// Threshold for egregious violations during warm-up (score below this = instant reject)
    pub egregious_threshold: f32,

    /// Base Byzantine detection threshold (quality below this = flagged)
    pub byzantine_threshold: f32,

    /// Minimum score for acceptance (after all adjustments)
    pub min_acceptance_score: f32,

    /// Enable direction-based prefilter
    pub direction_prefilter: bool,

    /// Cosine similarity threshold for direction prefilter
    pub direction_threshold: f32,
}

impl Default for PoGQv41Config {
    fn default() -> Self {
        Self {
            ema_beta: 0.85,
            warmup_rounds: 3,
            hysteresis_k: 2,
            hysteresis_m: 3,
            egregious_threshold: 0.15,
            byzantine_threshold: 0.5,
            min_acceptance_score: 0.3,
            direction_prefilter: true,
            direction_threshold: 0.0,
        }
    }
}

/// Client state for tracking warm-up and hysteresis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientState {
    /// EMA-smoothed score history
    pub ema_score: f32,

    /// Number of rounds this client has participated
    pub round_count: u32,

    /// Consecutive violation count
    pub consecutive_violations: u32,

    /// Consecutive clear count
    pub consecutive_clears: u32,

    /// Whether client is currently quarantined
    pub quarantined: bool,

    /// Last raw score
    pub last_raw_score: f32,

    /// Last round number
    pub last_round: u32,

    /// Historical scores for egregious cap calculation
    pub score_history: Vec<f32>,
}

impl ClientState {
    /// Create new client state with initial score
    pub fn new(initial_score: f32) -> Self {
        Self {
            ema_score: initial_score,
            round_count: 0, // Start at 0, will be incremented on first evaluate
            consecutive_violations: 0,
            consecutive_clears: 0,
            quarantined: false,
            last_raw_score: initial_score,
            last_round: 0,
            score_history: vec![initial_score],
        }
    }

    /// Check if client is in warm-up period
    pub fn in_warmup(&self, warmup_rounds: u32) -> bool {
        self.round_count <= warmup_rounds
    }
}

/// Result of PoGQ evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoGQEvaluation {
    /// Client ID
    pub client_id: String,

    /// Round number
    pub round: u32,

    /// Raw quality score before adjustments
    pub raw_score: f32,

    /// EMA-smoothed score
    pub ema_score: f32,

    /// Final adjusted score
    pub final_score: f32,

    /// Whether this gradient is Byzantine
    pub is_byzantine: bool,

    /// Whether client is quarantined
    pub is_quarantined: bool,

    /// Whether client is in warm-up period
    pub in_warmup: bool,

    /// Rejection reason (if Byzantine)
    pub rejection_reason: Option<String>,

    /// Confidence level of detection
    pub confidence: f32,
}

/// PoGQ v4.1 Enhanced Detector (Lite version for WASM)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoGQv41Enhanced {
    /// Configuration
    pub config: PoGQv41Config,

    /// Per-client state tracking
    client_states: HashMap<String, ClientState>,

    /// Global score statistics for egregious cap
    global_scores: Vec<f32>,

    /// Computed egregious cap (p99.9)
    egregious_cap: Option<f32>,

    /// Current round
    current_round: u32,
}

impl PoGQv41Enhanced {
    /// Create a new enhanced PoGQ detector
    pub fn new(config: PoGQv41Config) -> Self {
        Self {
            config,
            client_states: HashMap::new(),
            global_scores: Vec::new(),
            egregious_cap: None,
            current_round: 0,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PoGQv41Config::default())
    }

    /// Advance to next round
    pub fn next_round(&mut self) {
        self.current_round += 1;
    }

    /// Set current round
    pub fn set_round(&mut self, round: u32) {
        self.current_round = round;
    }

    /// Get current round
    pub fn round(&self) -> u32 {
        self.current_round
    }

    /// Evaluate a gradient contribution
    ///
    /// # Arguments
    /// * `client_id` - Unique client identifier
    /// * `quality` - Raw quality score (0.0-1.0)
    /// * `consistency` - Consistency with previous contributions (0.0-1.0)
    /// * `direction_score` - Optional cosine similarity with reference (for prefilter)
    ///
    /// # Returns
    /// Comprehensive evaluation result
    pub fn evaluate(
        &mut self,
        client_id: &str,
        quality: f32,
        consistency: f32,
        direction_score: Option<f32>,
    ) -> PoGQEvaluation {
        let round = self.current_round;

        // Direction prefilter: cheap early rejection
        if self.config.direction_prefilter {
            if let Some(dir) = direction_score {
                if dir < self.config.direction_threshold {
                    return PoGQEvaluation {
                        client_id: client_id.to_string(),
                        round,
                        raw_score: quality,
                        ema_score: 0.0,
                        final_score: 0.0,
                        is_byzantine: true,
                        is_quarantined: false,
                        in_warmup: false,
                        rejection_reason: Some(
                            "Direction prefilter: negative gradient direction".into(),
                        ),
                        confidence: 0.95,
                    };
                }
            }
        }

        // Compute raw score (weighted average of quality and consistency)
        let raw_score = 0.7 * quality + 0.3 * consistency;

        // Track global scores for egregious cap FIRST (before borrowing client_states)
        self.global_scores.push(raw_score);
        if self.global_scores.len() > 1000 {
            self.global_scores.remove(0);
        }

        // Compute egregious cap (p99.9 of score distribution inverted)
        self.update_egregious_cap();

        // Copy config values we need (to avoid borrowing self later)
        let config_ema_beta = self.config.ema_beta;
        let config_warmup_rounds = self.config.warmup_rounds;
        let config_hysteresis_k = self.config.hysteresis_k;
        let config_hysteresis_m = self.config.hysteresis_m;
        let egregious_cap = self.egregious_cap;
        let config_egregious_threshold = self.config.egregious_threshold;
        let config_byzantine_threshold = self.config.byzantine_threshold;

        // Get or create client state
        let state = self
            .client_states
            .entry(client_id.to_string())
            .or_insert_with(|| ClientState::new(raw_score));

        // Save previous score for sudden drop detection
        let prev_raw_score = state.last_raw_score;
        let prev_round_count = state.round_count;

        // Update EMA score
        let ema_score = config_ema_beta * state.ema_score + (1.0 - config_ema_beta) * raw_score;
        state.ema_score = ema_score;
        state.last_raw_score = raw_score;
        state.last_round = round;
        state.round_count += 1;

        // Check warm-up status
        let in_warmup = state.in_warmup(config_warmup_rounds);

        // Determine if Byzantine (inline to avoid borrow issues)
        let (is_byzantine, rejection_reason, confidence) = {
            // Egregious cap: instant reject even during warm-up
            if let Some(cap) = egregious_cap {
                if raw_score < cap {
                    (
                        true,
                        Some(format!(
                            "Egregious violation: score {:.3} < cap {:.3}",
                            raw_score, cap
                        )),
                        0.98,
                    )
                } else if in_warmup {
                    if raw_score < config_egregious_threshold {
                        (
                            true,
                            Some(format!(
                                "Warm-up egregious: score {:.3} < threshold {:.3}",
                                raw_score, config_egregious_threshold
                            )),
                            0.90,
                        )
                    } else {
                        (false, None, 0.0)
                    }
                } else if ema_score < config_byzantine_threshold {
                    let conf =
                        (config_byzantine_threshold - ema_score) / config_byzantine_threshold;
                    (
                        true,
                        Some(format!(
                            "Byzantine: EMA score {:.3} < threshold {:.3}",
                            ema_score, config_byzantine_threshold
                        )),
                        conf.clamp(0.5, 0.99),
                    )
                } else if prev_round_count > 1 {
                    let score_drop = prev_raw_score - raw_score;
                    if score_drop > 0.4 {
                        (
                            true,
                            Some(format!(
                                "Sudden score drop: {:.3} (from {:.3} to {:.3})",
                                score_drop, prev_raw_score, raw_score
                            )),
                            0.75,
                        )
                    } else {
                        (false, None, 0.0)
                    }
                } else {
                    (false, None, 0.0)
                }
            } else if in_warmup {
                if raw_score < config_egregious_threshold {
                    (
                        true,
                        Some(format!(
                            "Warm-up egregious: score {:.3} < threshold {:.3}",
                            raw_score, config_egregious_threshold
                        )),
                        0.90,
                    )
                } else {
                    (false, None, 0.0)
                }
            } else if ema_score < config_byzantine_threshold {
                let conf = (config_byzantine_threshold - ema_score) / config_byzantine_threshold;
                (
                    true,
                    Some(format!(
                        "Byzantine: EMA score {:.3} < threshold {:.3}",
                        ema_score, config_byzantine_threshold
                    )),
                    conf.clamp(0.5, 0.99),
                )
            } else if prev_round_count > 1 {
                // Check for sudden score drop (potential attack)
                let score_drop = prev_raw_score - raw_score;
                if score_drop > 0.4 {
                    (
                        true,
                        Some(format!(
                            "Sudden score drop: {:.3} (from {:.3} to {:.3})",
                            score_drop, prev_raw_score, raw_score
                        )),
                        0.75,
                    )
                } else {
                    (false, None, 0.0)
                }
            } else {
                (false, None, 0.0)
            }
        };

        // Update hysteresis state
        if is_byzantine {
            state.consecutive_violations += 1;
            state.consecutive_clears = 0;

            // Apply hysteresis: quarantine after k consecutive violations
            if state.consecutive_violations >= config_hysteresis_k && !state.quarantined {
                state.quarantined = true;
            }
        } else {
            state.consecutive_clears += 1;
            state.consecutive_violations = 0;

            // Release from quarantine after m consecutive clears
            if state.consecutive_clears >= config_hysteresis_m && state.quarantined {
                state.quarantined = false;
            }
        }

        // Track score history
        state.score_history.push(raw_score);
        if state.score_history.len() > 100 {
            state.score_history.remove(0);
        }

        // Final score considers EMA and quarantine
        let final_score = if state.quarantined {
            0.0 // Quarantined clients get zero score
        } else {
            ema_score
        };

        PoGQEvaluation {
            client_id: client_id.to_string(),
            round,
            raw_score,
            ema_score,
            final_score,
            is_byzantine: is_byzantine || state.quarantined,
            is_quarantined: state.quarantined,
            in_warmup,
            rejection_reason,
            confidence,
        }
    }

    /// Update egregious cap based on score distribution
    fn update_egregious_cap(&mut self) {
        if self.global_scores.len() < 10 {
            self.egregious_cap = None;
            return;
        }

        // Compute p0.1 (1 - p99.9) of scores as the cap
        let mut sorted = self.global_scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = (sorted.len() as f32 * 0.001).ceil() as usize;
        self.egregious_cap = Some(sorted[idx.min(sorted.len() - 1)]);
    }

    /// Get client state
    pub fn get_client_state(&self, client_id: &str) -> Option<&ClientState> {
        self.client_states.get(client_id)
    }

    /// Get all quarantined clients
    pub fn quarantined_clients(&self) -> Vec<String> {
        self.client_states
            .iter()
            .filter(|(_, state)| state.quarantined)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get detection statistics
    pub fn statistics(&self) -> DetectionStatistics {
        let total = self.client_states.len();
        let quarantined = self.quarantined_clients().len();
        let in_warmup = self
            .client_states
            .values()
            .filter(|s| s.in_warmup(self.config.warmup_rounds))
            .count();

        let avg_score = if self.global_scores.is_empty() {
            0.0
        } else {
            self.global_scores.iter().sum::<f32>() / self.global_scores.len() as f32
        };

        DetectionStatistics {
            total_clients: total,
            quarantined_clients: quarantined,
            clients_in_warmup: in_warmup,
            average_score: avg_score,
            egregious_cap: self.egregious_cap,
            current_round: self.current_round,
        }
    }

    /// Reset all client states (for testing)
    pub fn reset(&mut self) {
        self.client_states.clear();
        self.global_scores.clear();
        self.egregious_cap = None;
        self.current_round = 0;
    }
}

/// Detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionStatistics {
    /// Total number of tracked clients
    pub total_clients: usize,
    /// Number of currently quarantined clients
    pub quarantined_clients: usize,
    /// Number of clients still in warm-up period
    pub clients_in_warmup: usize,
    /// Average raw score across all recent evaluations
    pub average_score: f32,
    /// Computed egregious cap (p0.1 of score distribution)
    pub egregious_cap: Option<f32>,
    /// Current round number
    pub current_round: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_grace_period() {
        let mut detector = PoGQv41Enhanced::with_defaults();

        // During warm-up, even lower scores (above egregious threshold) should pass
        // raw_score = 0.7*0.35 + 0.3*0.35 = 0.35 (below byzantine threshold of 0.5)
        for round in 0..3 {
            detector.set_round(round);
            let eval = detector.evaluate("client_1", 0.35, 0.35, None);
            assert!(
                !eval.is_byzantine,
                "Round {}: Should not reject during warm-up (score above egregious)",
                round
            );
            assert!(eval.in_warmup, "Round {}: Should be in warm-up", round);
        }

        // After warm-up, same score should be flagged (EMA below threshold)
        detector.set_round(4);
        let eval = detector.evaluate("client_1", 0.35, 0.35, None);
        assert!(
            eval.is_byzantine,
            "Should reject after warm-up (EMA below threshold)"
        );
        assert!(!eval.in_warmup, "Should not be in warm-up");
    }

    #[test]
    fn test_hysteresis_quarantine() {
        let mut detector = PoGQv41Enhanced::with_defaults();
        detector.config.warmup_rounds = 0; // Disable warm-up for test
        detector.config.hysteresis_k = 2; // 2 violations to quarantine
        detector.config.hysteresis_m = 3; // 3 clears to release
        detector.config.ema_beta = 0.5; // Faster EMA for testing
        detector.config.byzantine_threshold = 0.4;

        // Start with a good score to initialize
        let eval = detector.evaluate("client_1", 0.8, 0.8, None);
        assert!(!eval.is_quarantined);
        assert!(!eval.is_byzantine);

        // Send low scores to drop EMA below threshold
        // With beta=0.5, EMA drops faster
        for _ in 0..5 {
            detector.evaluate("client_1", 0.1, 0.1, None);
        }

        // Now EMA should be below threshold, consecutive violations should accumulate
        // First Byzantine (EMA below threshold)
        let eval = detector.evaluate("client_1", 0.1, 0.1, None);
        assert!(
            eval.is_byzantine,
            "Should be Byzantine (EMA below threshold)"
        );
        // Check consecutive_violations via is_quarantined (k=2)
        // After first Byzantine post-threshold, consecutive_violations=1

        // Second Byzantine - should trigger quarantine
        let eval = detector.evaluate("client_1", 0.1, 0.1, None);
        assert!(eval.is_byzantine);
        assert!(
            eval.is_quarantined,
            "Should be quarantined after k=2 violations"
        );

        // Even good scores won't immediately release (m=3 needed)
        for i in 0..2 {
            let eval = detector.evaluate("client_1", 0.9, 0.9, None);
            assert!(
                eval.is_quarantined,
                "Round {}: Still quarantined (need 3 clears)",
                i
            );
        }

        // Third clear should release
        let eval = detector.evaluate("client_1", 0.9, 0.9, None);
        assert!(!eval.is_quarantined, "Should be released after m=3 clears");
    }

    #[test]
    fn test_ema_smoothing() {
        let mut detector = PoGQv41Enhanced::with_defaults();
        detector.config.warmup_rounds = 0;

        // Establish history
        for _ in 0..5 {
            detector.evaluate("client_1", 0.8, 0.8, None);
        }

        // Single low score should not immediately flag due to EMA
        let eval = detector.evaluate("client_1", 0.3, 0.3, None);
        // EMA should be around 0.85 * 0.75 + 0.15 * 0.3 = 0.68
        assert!(eval.ema_score > 0.5);
    }

    #[test]
    fn test_egregious_cap() {
        let mut detector = PoGQv41Enhanced::with_defaults();

        // Build up score history
        for i in 0..50 {
            detector.evaluate(
                &format!("client_{}", i),
                0.7 + (i as f32 % 3.0) * 0.1,
                0.8,
                None,
            );
        }

        // Egregious score should be rejected even during warm-up
        let eval = detector.evaluate("new_client", 0.01, 0.01, None);
        assert!(eval.is_byzantine);
        assert!(eval.rejection_reason.is_some());
    }

    #[test]
    fn test_direction_prefilter() {
        let mut detector = PoGQv41Enhanced::with_defaults();

        // Negative direction should be rejected immediately
        let eval = detector.evaluate("client_1", 0.9, 0.9, Some(-0.5));
        assert!(eval.is_byzantine);
        assert!(eval
            .rejection_reason
            .unwrap()
            .contains("Direction prefilter"));
    }
}
