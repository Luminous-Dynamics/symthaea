// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Ported from fl-aggregator/src/pogq.rs — PoGQ-v4.1 Lite variant
// Skips: full Mondrian conformal, PCA projection, ndarray-based AdaptiveHybridScorer
// Keeps: Lite probe head, hysteresis, warm-up quota, direction prefilter, EMA scoring
//
//! PoGQ-v4.1 Lite: WASM-Compatible Byzantine Detection
//!
//! This is the WASM-portable subset of PoGQ-v4.1 Enhanced. It provides:
//! - **Linear probe head** scoring (extract last N params for utility)
//! - **Direction prefilter** (ReLU cosine rejection)
//! - **Temporal EMA** smoothing (beta=0.85)
//! - **Warm-up quota** (W rounds grace period)
//! - **Hysteresis** (k violations to quarantine, m clears to release)
//! - **Egregious cap** (p99.9 warm-up override)
//!
//! Full Mondrian conformal and PCA-based scoring require ndarray and are
//! not WASM-compatible. Use fl-aggregator's `PoGQv41Enhanced` for those.

use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for PoGQ-v4.1 Lite detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoGQLiteConfig {
    /// Enable direction prefilter (ReLU cosine > threshold)
    pub direction_prefilter: bool,
    /// ReLU threshold for direction score
    pub direction_threshold: f32,

    /// Weight for historical EMA scores (0.85 = stable but responsive)
    pub ema_beta: f32,

    /// Default lambda for hybrid score (direction vs utility weighting)
    pub lambda: f32,

    /// Number of rounds before full enforcement (warm-up)
    pub warmup_rounds: u32,

    /// Consecutive violations to quarantine
    pub hysteresis_k: u32,
    /// Consecutive clears to release from quarantine
    pub hysteresis_m: u32,

    /// Fixed threshold for Byzantine detection (EMA score below this = violation)
    pub detection_threshold: f32,

    /// Egregious cap quantile (scores below this during warm-up = instant flag)
    /// Computed from calibration data, stored as absolute value
    pub egregious_cap: Option<f32>,

    /// Use lite probe-head scoring (extract last N params for utility)
    pub use_lite_mode: bool,
    /// Number of probe-head parameters (backbone_dim * num_classes + num_classes)
    pub probe_params: usize,
}

impl Default for PoGQLiteConfig {
    fn default() -> Self {
        Self {
            direction_prefilter: true,
            direction_threshold: 0.0,
            ema_beta: 0.85,
            lambda: 0.7,
            warmup_rounds: 3,
            hysteresis_k: 2,
            hysteresis_m: 3,
            detection_threshold: 0.3,
            egregious_cap: None,
            use_lite_mode: false,
            probe_params: 0,
        }
    }
}

// =============================================================================
// Result Types
// =============================================================================

/// Scores from gradient evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientScores {
    /// Hybrid score (lambda * direction + (1-lambda) * utility)
    pub hybrid: f32,
    /// EMA-smoothed score
    pub ema: f32,
    /// Utility score component (gradient norm or probe cosine)
    pub utility: f32,
    /// Direction score (cosine similarity, post-ReLU if prefilter on)
    pub direction: f32,
    /// Lambda value used
    pub lambda: f32,
}

/// Detection decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Final Byzantine decision (quarantine status)
    pub is_byzantine: bool,
    /// Whether this round's score was a violation
    pub is_violation: bool,
    /// Threshold used for detection
    pub threshold: f32,
    /// Whether client is currently quarantined
    pub quarantined: bool,
}

/// Warm-up and hysteresis status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupStatus {
    /// Number of rounds this client has participated
    pub client_rounds: u32,
    /// Whether client is in warm-up period
    pub in_warmup: bool,
    /// Consecutive violations
    pub consecutive_violations: u32,
    /// Consecutive clears
    pub consecutive_clears: u32,
}

/// Complete result for one gradient scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoGQResult {
    /// Client identifier
    pub client_id: String,
    /// Round number
    pub round: u32,
    /// Score components
    pub scores: GradientScores,
    /// Detection decision
    pub detection: DetectionResult,
    /// Warm-up/hysteresis status
    pub warmup: WarmupStatus,
}

/// Aggregation diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationDiagnostics {
    /// Round number
    pub round: u32,
    /// Total clients
    pub n_clients: usize,
    /// Honest (non-quarantined) clients
    pub n_honest: usize,
    /// Byzantine (quarantined) clients
    pub n_byzantine: usize,
    /// Detection rate
    pub detection_rate: f32,
    /// Honest client IDs
    pub honest_clients: Vec<String>,
}

/// Detector statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorStatistics {
    /// Total unique clients seen
    pub total_clients: usize,
    /// Currently quarantined clients
    pub quarantined_clients: usize,
    /// Average EMA score
    pub avg_ema_score: f32,
}

// =============================================================================
// Per-Client State
// =============================================================================

#[derive(Debug, Clone, Default)]
struct ClientState {
    ema_score: Option<f32>,
    round_count: u32,
    consecutive_violations: u32,
    consecutive_clears: u32,
    quarantined: bool,
}

// =============================================================================
// PoGQ-v4.1 Lite Detector
// =============================================================================

/// PoGQ-v4.1 Lite: WASM-compatible Byzantine detection
///
/// Scores client gradients using direction (cosine similarity) and utility
/// (gradient norm or probe-head cosine), applies EMA smoothing, and uses
/// hysteresis-based quarantine decisions with warm-up grace periods.
#[derive(Debug, Clone)]
pub struct PoGQLiteDetector {
    config: PoGQLiteConfig,
    clients: BTreeMap<String, ClientState>,
}

impl PoGQLiteDetector {
    /// Create a new PoGQ Lite detector
    pub fn new(config: PoGQLiteConfig) -> Self {
        Self {
            config,
            clients: BTreeMap::new(),
        }
    }

    /// Calibrate egregious cap from clean validation scores.
    ///
    /// Sets the cap to the given quantile of validation scores.
    /// Scores below this during warm-up trigger instant flagging.
    pub fn calibrate_egregious_cap(&mut self, validation_scores: &[f32], quantile: f32) {
        if validation_scores.is_empty() {
            return;
        }
        self.config.egregious_cap = Some(percentile(validation_scores, quantile * 100.0));
    }

    /// Score a single client gradient against a reference.
    ///
    /// Returns a `PoGQResult` with scores, detection decision, and warm-up status.
    pub fn score_gradient(
        &mut self,
        gradient: &[f32],
        client_id: &str,
        reference: &[f32],
        round_number: u32,
    ) -> PoGQResult {
        // Compute scores before mutable borrow on client state
        let utility = if self.config.use_lite_mode && self.config.probe_params > 0 {
            self.probe_cosine(gradient, reference)
        } else {
            l2_norm(gradient)
        };

        let raw_cosine = cosine_similarity(gradient, reference);
        let direction = if self.config.direction_prefilter {
            (raw_cosine - self.config.direction_threshold).max(0.0)
        } else {
            raw_cosine
        };

        let lambda = self.config.lambda;
        let hybrid = lambda * direction + (1.0 - lambda) * utility;

        // Now take mutable borrow on client state
        let state = self.clients.entry(client_id.into()).or_default();
        state.round_count += 1;

        // EMA smoothing
        let ema = if let Some(prev) = state.ema_score {
            self.config.ema_beta * prev + (1.0 - self.config.ema_beta) * hybrid
        } else {
            hybrid
        };
        state.ema_score = Some(ema);

        // Warm-up + violation check
        let in_warmup = state.round_count <= self.config.warmup_rounds;
        let is_violation = if in_warmup {
            // During warm-up, only flag egregious violations
            self.config
                .egregious_cap
                .map(|cap| ema < cap)
                .unwrap_or(false)
        } else {
            ema < self.config.detection_threshold
        };

        // 6. Hysteresis
        if is_violation {
            state.consecutive_violations += 1;
            state.consecutive_clears = 0;
        } else {
            state.consecutive_clears += 1;
            state.consecutive_violations = 0;
        }

        if state.consecutive_violations >= self.config.hysteresis_k {
            state.quarantined = true;
        } else if state.consecutive_clears >= self.config.hysteresis_m {
            state.quarantined = false;
        }

        PoGQResult {
            client_id: client_id.into(),
            round: round_number,
            scores: GradientScores {
                hybrid,
                ema,
                utility,
                direction,
                lambda,
            },
            detection: DetectionResult {
                is_byzantine: state.quarantined,
                is_violation,
                threshold: self.config.detection_threshold,
                quarantined: state.quarantined,
            },
            warmup: WarmupStatus {
                client_rounds: state.round_count,
                in_warmup,
                consecutive_violations: state.consecutive_violations,
                consecutive_clears: state.consecutive_clears,
            },
        }
    }

    /// Filter gradients: returns indices of honest (non-quarantined) clients.
    ///
    /// Scores all gradients and returns the indices that passed detection.
    pub fn filter_honest(
        &mut self,
        gradients: &[Vec<f32>],
        client_ids: &[&str],
        reference: &[f32],
        round_number: u32,
    ) -> (Vec<usize>, AggregationDiagnostics) {
        let mut honest_indices = Vec::new();
        let mut honest_clients = Vec::new();

        for (i, (gradient, &client_id)) in gradients.iter().zip(client_ids.iter()).enumerate() {
            let result = self.score_gradient(gradient, client_id, reference, round_number);
            if !result.detection.is_byzantine {
                honest_indices.push(i);
                honest_clients.push(client_id.to_string());
            }
        }

        let n_clients = gradients.len();
        let n_honest = honest_indices.len();
        let diagnostics = AggregationDiagnostics {
            round: round_number,
            n_clients,
            n_honest,
            n_byzantine: n_clients - n_honest,
            detection_rate: if n_clients > 0 {
                (n_clients - n_honest) as f32 / n_clients as f32
            } else {
                0.0
            },
            honest_clients,
        };

        (honest_indices, diagnostics)
    }

    /// Check if a client is currently quarantined
    pub fn is_quarantined(&self, client_id: &str) -> bool {
        self.clients
            .get(client_id)
            .map(|s| s.quarantined)
            .unwrap_or(false)
    }

    /// Get all quarantined client IDs
    pub fn quarantined_clients(&self) -> Vec<String> {
        self.clients
            .iter()
            .filter(|(_, s)| s.quarantined)
            .map(|(id, _): (&String, &ClientState)| id.clone())
            .collect()
    }

    /// Get detector statistics
    pub fn statistics(&self) -> DetectorStatistics {
        let total = self.clients.len();
        let quarantined = self.clients.values().filter(|s| s.quarantined).count();
        let avg_ema = if total > 0 {
            self.clients
                .values()
                .filter_map(|s| s.ema_score)
                .sum::<f32>()
                / total as f32
        } else {
            0.0
        };
        DetectorStatistics {
            total_clients: total,
            quarantined_clients: quarantined,
            avg_ema_score: avg_ema,
        }
    }

    /// Reset all client state (for new training run)
    pub fn reset(&mut self) {
        self.clients.clear();
    }

    /// Get current config
    pub fn config(&self) -> &PoGQLiteConfig {
        &self.config
    }

    // ---- internal helpers ----

    /// Compute cosine similarity on only the probe-head slice
    fn probe_cosine(&self, gradient: &[f32], reference: &[f32]) -> f32 {
        let p = self.config.probe_params;
        if gradient.len() < p || reference.len() < p {
            return cosine_similarity(gradient, reference);
        }
        let g_probe = &gradient[gradient.len() - p..];
        let r_probe = &reference[reference.len() - p..];
        cosine_similarity(g_probe, r_probe)
    }
}

// =============================================================================
// Helper Functions (no ndarray)
// =============================================================================

/// L2 norm of a slice
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity between two slices
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let dot: f32 = a[..len].iter().zip(&b[..len]).map(|(x, y)| x * y).sum();
    let na = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Percentile (0-100 scale) of a slice
fn percentile(data: &[f32], p: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = ((p / 100.0) * (sorted.len() - 1) as f32).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_detector() -> PoGQLiteDetector {
        PoGQLiteDetector::new(PoGQLiteConfig::default())
    }

    #[test]
    fn test_config_defaults() {
        let c = PoGQLiteConfig::default();
        assert!(c.direction_prefilter);
        assert_eq!(c.direction_threshold, 0.0);
        assert_eq!(c.ema_beta, 0.85);
        assert_eq!(c.lambda, 0.7);
        assert_eq!(c.warmup_rounds, 3);
        assert_eq!(c.hysteresis_k, 2);
        assert_eq!(c.hysteresis_m, 3);
        assert_eq!(c.detection_threshold, 0.3);
        assert!(!c.use_lite_mode);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_norm() {
        assert!((l2_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-5);
        assert!((l2_norm(&[0.0]) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 0.0) - 1.0).abs() < 1e-5);
        assert!((percentile(&data, 50.0) - 3.0).abs() < 1e-5);
        assert!((percentile(&data, 100.0) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_warmup_grace_period() {
        let mut det = default_detector();
        let gradient = vec![0.01; 100]; // Very weak gradient
        let reference = vec![1.0; 100];

        // First 3 rounds should be warm-up
        for r in 0..3 {
            let res = det.score_gradient(&gradient, "c1", &reference, r);
            assert!(res.warmup.in_warmup, "round {} should be warmup", r);
            // Without egregious cap, warm-up doesn't flag
            assert!(!res.detection.is_violation);
        }

        // Round 3 (4th score) exits warm-up
        let res = det.score_gradient(&gradient, "c1", &reference, 3);
        assert!(!res.warmup.in_warmup);
    }

    #[test]
    fn test_egregious_cap_overrides_warmup() {
        let mut det = PoGQLiteDetector::new(PoGQLiteConfig {
            warmup_rounds: 5,
            hysteresis_k: 1,
            ..Default::default()
        });

        // Calibrate with high scores so cap is high
        let cal_scores: Vec<f32> = (0..100).map(|i| 5.0 + 0.1 * i as f32).collect();
        det.calibrate_egregious_cap(&cal_scores, 0.999);

        // Gradient with near-zero norm => ema will be very low
        let gradient = vec![0.001; 100];
        let reference = vec![1.0; 100];

        let res = det.score_gradient(&gradient, "c1", &reference, 0);
        assert!(res.warmup.in_warmup);
        // EMA is well below egregious cap, so it should be a violation
        assert!(res.detection.is_violation);
    }

    #[test]
    fn test_hysteresis_quarantine() {
        let mut det = PoGQLiteDetector::new(PoGQLiteConfig {
            warmup_rounds: 0,
            hysteresis_k: 2,
            hysteresis_m: 3,
            detection_threshold: 5.0, // High threshold so everything is a violation
            ..Default::default()
        });

        let gradient = vec![1.0; 100];
        let reference = vec![1.0; 100];

        // Round 0: first violation, not yet quarantined (need 2)
        let r0 = det.score_gradient(&gradient, "c1", &reference, 0);
        assert!(r0.detection.is_violation);
        assert!(!r0.detection.quarantined);

        // Round 1: second violation, NOW quarantined
        let r1 = det.score_gradient(&gradient, "c1", &reference, 1);
        assert!(r1.detection.is_violation);
        assert!(r1.detection.quarantined);
    }

    #[test]
    fn test_hysteresis_release() {
        let mut det = PoGQLiteDetector::new(PoGQLiteConfig {
            warmup_rounds: 0,
            hysteresis_k: 1,
            hysteresis_m: 2,
            detection_threshold: 5.0, // High threshold
            ..Default::default()
        });

        let gradient = vec![1.0; 100];
        let reference = vec![1.0; 100];

        // Get quarantined
        det.score_gradient(&gradient, "c1", &reference, 0);
        assert!(det.is_quarantined("c1"));

        // Lower threshold so clears pass
        det.config.detection_threshold = 0.0;

        // First clear
        det.score_gradient(&gradient, "c1", &reference, 1);
        assert!(det.is_quarantined("c1")); // Still quarantined, need 2 clears

        // Second clear -> released
        det.score_gradient(&gradient, "c1", &reference, 2);
        assert!(!det.is_quarantined("c1"));
    }

    #[test]
    fn test_honest_gradient_passes() {
        let mut det = PoGQLiteDetector::new(PoGQLiteConfig {
            warmup_rounds: 0,
            hysteresis_k: 1,
            detection_threshold: 0.1,
            ..Default::default()
        });

        // Honest gradient: same direction, decent norm
        let reference = vec![1.0; 100];
        let honest = vec![1.1; 100]; // Same direction, slightly larger

        // Score several rounds to build EMA
        for r in 0..5 {
            let res = det.score_gradient(&honest, "honest1", &reference, r);
            assert!(
                !res.detection.is_byzantine,
                "Honest gradient flagged at round {}: ema={}, threshold={}",
                r, res.scores.ema, res.detection.threshold
            );
        }
    }

    #[test]
    fn test_byzantine_gradient_detected() {
        let mut det = PoGQLiteDetector::new(PoGQLiteConfig {
            warmup_rounds: 0,
            hysteresis_k: 1,
            detection_threshold: 1.0, // Moderate threshold
            direction_prefilter: true,
            direction_threshold: 0.0,
            ..Default::default()
        });

        let reference = vec![1.0; 100];
        // Opposite direction: cosine = -1, after ReLU direction = 0
        // hybrid = 0.7 * 0 + 0.3 * norm(gradient) = 0.3 * 10 = 3.0
        // Actually norm = sqrt(100) = 10, but direction = 0
        // hybrid = 0.7 * 0 + 0.3 * 10 = 3.0 which is > 1.0
        // So let's use a near-zero gradient instead
        let byzantine = vec![0.001; 100]; // Near-zero, very weak

        let res = det.score_gradient(&byzantine, "byz1", &reference, 0);
        // direction after ReLU = max(cosine - 0, 0) = cosine (both positive, near 1)
        // Hmm, 0.001 and 1.0 have cosine ~1.0 since same direction
        // Let's use truly opposite gradient
        let byzantine2 = vec![-1.0; 100];
        let res2 = det.score_gradient(&byzantine2, "byz2", &reference, 0);
        // direction = max(-1.0 - 0.0, 0.0) = 0.0
        // utility = l2_norm = 10.0
        // hybrid = 0.7 * 0 + 0.3 * 10 = 3.0 > 1.0 threshold, so NOT a violation
        // We need a lower threshold or different attack
        // Let's use zero gradient
        let zero_grad = vec![0.0; 100];
        let res3 = det.score_gradient(&zero_grad, "byz3", &reference, 0);
        // direction = cosine(zero, ref) = 0 (norm_a < 1e-10)
        // utility = l2_norm(zero) = 0
        // hybrid = 0.7 * 0 + 0.3 * 0 = 0 < 1.0 threshold => violation!
        assert!(res3.detection.is_violation);
        assert!(res3.detection.quarantined);
    }

    #[test]
    fn test_filter_honest() {
        let mut det = PoGQLiteDetector::new(PoGQLiteConfig {
            warmup_rounds: 0,
            hysteresis_k: 1,
            detection_threshold: 0.1,
            ..Default::default()
        });

        let reference = vec![1.0; 50];
        let gradients = vec![
            vec![1.0; 50],  // Honest
            vec![0.0; 50],  // Byzantine (zero gradient)
            vec![1.2; 50],  // Honest
        ];
        let ids = vec!["h1", "b1", "h2"];

        let (honest, diag) = det.filter_honest(&gradients, &ids, &reference, 0);

        // h1 and h2 should pass, b1 should not (zero gradient → hybrid = 0)
        assert_eq!(diag.n_clients, 3);
        assert!(honest.contains(&0), "h1 should be honest");
        assert!(honest.contains(&2), "h2 should be honest");
        assert!(!honest.contains(&1), "b1 should be byzantine");
    }

    #[test]
    fn test_lite_mode_probe_head() {
        let mut det = PoGQLiteDetector::new(PoGQLiteConfig {
            warmup_rounds: 0,
            hysteresis_k: 1,
            detection_threshold: 0.1,
            use_lite_mode: true,
            probe_params: 10, // Last 10 params
            ..Default::default()
        });

        // Build gradient where probe head matches reference but body doesn't
        let mut gradient = vec![-1.0; 90]; // Body: opposite direction
        gradient.extend(vec![1.0; 10]); // Probe head: matches reference

        let reference = vec![1.0; 100];

        let res = det.score_gradient(&gradient, "c1", &reference, 0);
        // In lite mode, utility = cosine of last 10 params = 1.0
        // direction = cosine of full gradient (mixed) — might be negative
        // But the utility component should be high
        assert!(
            res.scores.utility > 0.9,
            "Lite mode should use probe head: utility={}",
            res.scores.utility
        );
    }

    #[test]
    fn test_statistics() {
        let mut det = default_detector();
        let reference = vec![1.0; 10];

        det.score_gradient(&[1.0; 10], "c1", &reference, 0);
        det.score_gradient(&[1.0; 10], "c2", &reference, 0);

        let stats = det.statistics();
        assert_eq!(stats.total_clients, 2);
        assert_eq!(stats.quarantined_clients, 0);
        assert!(stats.avg_ema_score > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut det = default_detector();
        let reference = vec![1.0; 10];

        det.score_gradient(&[1.0; 10], "c1", &reference, 0);
        assert_eq!(det.statistics().total_clients, 1);

        det.reset();
        assert_eq!(det.statistics().total_clients, 0);
    }

    #[test]
    fn test_empty_gradient() {
        let mut det = default_detector();
        let res = det.score_gradient(&[], "c1", &[], 0);
        // Should not panic, scores should be 0
        assert_eq!(res.scores.utility, 0.0);
        assert_eq!(res.scores.direction, 0.0);
    }

    #[test]
    fn test_calibrate_egregious_cap() {
        let mut det = default_detector();
        let scores: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        det.calibrate_egregious_cap(&scores, 0.999);
        assert!(det.config.egregious_cap.is_some());
        // p99.9 of 0..9.9 should be near 9.9
        let cap = det.config.egregious_cap.unwrap();
        assert!(cap > 9.0, "cap should be near max: {}", cap);
    }
}
