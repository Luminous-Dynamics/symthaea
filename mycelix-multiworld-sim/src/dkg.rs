// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Distributed Key Generation (DKG) cost model for governance throughput.
//!
//! Phase 3 of the Mycelix simulation roadmap (survey Gap 3). The real
//! Feldman VSS implementation lives in `mycelix-core/libs/feldman-dkg/`
//! — this module doesn't replay the cryptography. It models the *cost*
//! side: how long a ceremony takes, how many messages it exchanges, and
//! how likely it is to succeed under dropout.
//!
//! ## Why this matters for the sim
//!
//! Mycelix's governance uses Feldman DKG for threshold signing on
//! high-stakes proposals (constitutional amendments, treasury movements,
//! emergency powers). Without a cost model, the sim treats ratification
//! as instantaneous. That masks a real-world tradeoff:
//!
//! - **Small committee (≤ 10)**: fast ceremony, low dropout tolerance,
//!   but few eyes on the decision.
//! - **Large committee (≥ 100)**: deep legitimacy, but O(N²) messages
//!   and stretch latency mean the ceremony may not complete under even
//!   modest participant dropout.
//!
//! The functions here let a scenario answer "how big can a committee
//! be before constitutional amendments stop ratifying in time?"
//!
//! ## Model simplifications
//!
//! - Two rounds: commitment distribution + verification/share exchange.
//!   Feldman VSS is technically 2 rounds + a finalization; we fold the
//!   finalization into round 2.
//! - Each round's latency = `rtt_ticks` (one round-trip of messages).
//! - Each participant sends to every other: `N × (N - 1)` messages per
//!   round, `2 × N × (N - 1)` total.
//! - Dropout is independent per participant per ceremony. Success
//!   requires ≥ `threshold` participants to survive both rounds.
//! - Verification time is folded into `rtt_ticks`.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for a DKG ceremony in the simulator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DkgConfig {
    /// Number of participants in the committee.
    pub committee_size: u32,
    /// Minimum number of participants required to finalize (t-of-n).
    pub threshold: u32,
    /// One-way round-trip time, in sim ticks. For a monthly-tick sim a
    /// ceremony normally completes within a single tick — `rtt_ticks = 0`
    /// means the ceremony is intra-tick; `rtt_ticks ≥ 1` means the
    /// ceremony spans ticks.
    pub rtt_ticks: u32,
    /// Per-participant probability of dropping out during the ceremony,
    /// in [0, 1].
    pub dropout_rate: f64,
}

impl DkgConfig {
    /// Construct a new config, clamping values to sane ranges. `threshold`
    /// is floored at 1 and capped at `committee_size`.
    pub fn new(committee_size: u32, threshold: u32, rtt_ticks: u32, dropout_rate: f64) -> Self {
        let cs = committee_size.max(1);
        Self {
            committee_size: cs,
            threshold: threshold.clamp(1, cs),
            rtt_ticks,
            dropout_rate: dropout_rate.clamp(0.0, 1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Cost + success estimates
// ---------------------------------------------------------------------------

/// A cost estimate for one DKG ceremony.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DkgCost {
    /// Latency in sim ticks (always ≥ 0).
    pub latency_ticks: u32,
    /// Total messages exchanged across both rounds.
    pub message_count: u64,
    /// Probability that at least `threshold` participants complete both
    /// rounds, in [0, 1]. Computed from the binomial distribution —
    /// a participant survives both rounds with probability
    /// `(1 − dropout)²`, and the ceremony succeeds if ≥ threshold survive.
    pub success_prob: f64,
}

impl DkgCost {
    /// Estimate cost for the given config.
    ///
    /// - `latency_ticks = 2 × rtt_ticks` (two rounds).
    /// - `message_count = 2 × N × (N − 1)` (all-to-all, both rounds).
    /// - `success_prob` = P(X ≥ t) where X ~ Binomial(N, (1 − dropout)²).
    pub fn estimate(cfg: &DkgConfig) -> Self {
        let n = cfg.committee_size as u64;
        let latency_ticks = cfg.rtt_ticks.saturating_mul(2);
        let message_count = 2 * n * n.saturating_sub(1);
        let per_survive = (1.0 - cfg.dropout_rate).powi(2);
        let success_prob = binomial_tail_ge(cfg.committee_size, cfg.threshold, per_survive);
        Self {
            latency_ticks,
            message_count,
            success_prob,
        }
    }

    /// Whether this ceremony is viable under the given reliability floor.
    pub fn is_viable(&self, floor: f64) -> bool {
        self.success_prob >= floor.clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Binomial tail probability
// ---------------------------------------------------------------------------

/// `P(X ≥ k)` where `X ~ Binomial(n, p)`. Returns 0.0 on pathological input.
/// Uses direct summation of probability-mass terms with a log-scale
/// intermediate for numerical stability on large `n`.
fn binomial_tail_ge(n: u32, k: u32, p: f64) -> f64 {
    if n == 0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if k == 0 {
        return 1.0;
    }
    if k > n {
        return 0.0;
    }
    if !(0.0..=1.0).contains(&p) {
        return 0.0;
    }
    // Special-case p = 1.0 to avoid log(0).
    if p >= 1.0 {
        return 1.0;
    }
    if p <= 0.0 {
        return 0.0;
    }

    // Sum PMF from k to n in log-space for stability.
    let ln_p = p.ln();
    let ln_q = (1.0 - p).ln();
    let mut total = 0.0_f64;
    // log C(n, i) computed incrementally: log C(n, i+1) = log C(n, i) + ln(n-i) - ln(i+1).
    // Start at i = k, so precompute log C(n, k).
    let mut log_c = log_binom_coeff(n, k);
    for i in k..=n {
        let log_pmf = log_c + (i as f64) * ln_p + ((n - i) as f64) * ln_q;
        total += log_pmf.exp();
        if i < n {
            // Advance to next term.
            log_c += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
        }
    }
    total.clamp(0.0, 1.0)
}

fn log_binom_coeff(n: u32, k: u32) -> f64 {
    if k == 0 || k == n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut log_c = 0.0_f64;
    for i in 0..k {
        log_c += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
    }
    log_c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_clamps_threshold() {
        let cfg = DkgConfig::new(5, 10, 1, 0.05);
        assert_eq!(cfg.threshold, 5);
        let cfg = DkgConfig::new(5, 0, 1, 0.05);
        assert_eq!(cfg.threshold, 1);
    }

    #[test]
    fn config_clamps_dropout_rate() {
        let cfg = DkgConfig::new(5, 3, 1, 2.0);
        assert_eq!(cfg.dropout_rate, 1.0);
        let cfg = DkgConfig::new(5, 3, 1, -0.5);
        assert_eq!(cfg.dropout_rate, 0.0);
    }

    #[test]
    fn latency_is_two_rounds() {
        let cfg = DkgConfig::new(5, 3, 1, 0.0);
        let cost = DkgCost::estimate(&cfg);
        assert_eq!(cost.latency_ticks, 2);
    }

    #[test]
    fn message_count_is_n_squared() {
        let cfg = DkgConfig::new(10, 7, 1, 0.0);
        let cost = DkgCost::estimate(&cfg);
        // 2 × 10 × 9 = 180
        assert_eq!(cost.message_count, 180);
    }

    #[test]
    fn zero_dropout_always_succeeds() {
        let cfg = DkgConfig::new(50, 34, 1, 0.0);
        let cost = DkgCost::estimate(&cfg);
        assert!((cost.success_prob - 1.0).abs() < 1e-9);
    }

    #[test]
    fn full_dropout_never_succeeds() {
        let cfg = DkgConfig::new(50, 34, 1, 1.0);
        let cost = DkgCost::estimate(&cfg);
        assert!(cost.success_prob < 1e-9);
    }

    #[test]
    fn small_committee_tolerates_dropout() {
        // 5-of-7, 5% dropout per participant, per_survive ≈ 0.9025.
        // P(X ≥ 5) where X ~ Bin(7, 0.9025) ≈ 0.96
        let cfg = DkgConfig::new(7, 5, 1, 0.05);
        let cost = DkgCost::estimate(&cfg);
        assert!(
            cost.success_prob > 0.9,
            "small committee should be robust: {}",
            cost.success_prob,
        );
        assert!(cost.is_viable(0.9));
    }

    #[test]
    fn large_committee_under_dropout_is_fragile() {
        // 70-of-100, 15% dropout per participant, per_survive = 0.7225.
        // Expected survivors: 72.25. Probability of ≥ 70 survivors: moderate.
        let cfg = DkgConfig::new(100, 70, 1, 0.15);
        let cost = DkgCost::estimate(&cfg);
        assert!(
            cost.success_prob < 0.8,
            "large committee + 15% dropout should be fragile: {}",
            cost.success_prob,
        );
        // Message count is 2 × 100 × 99 = 19,800 — three orders of magnitude
        // more than a 7-member committee.
        assert_eq!(cost.message_count, 19_800);
    }

    #[test]
    fn viability_respects_floor() {
        let cfg = DkgConfig::new(7, 5, 1, 0.05);
        let cost = DkgCost::estimate(&cfg);
        assert!(cost.is_viable(0.5));
        assert!(cost.is_viable(0.9));
        // At 99% floor even robust small committees may fall short.
        // The exact threshold depends on parameters — we just check the
        // method responds to the floor.
        if cost.success_prob < 0.99 {
            assert!(!cost.is_viable(0.99));
        }
    }

    #[test]
    fn committee_size_sweep_shows_cost_growth() {
        // Sweep 5, 20, 50, 100 at 2/3 threshold, 10% dropout.
        // Message count grows quadratically; latency is constant; success
        // probability is non-monotonic due to integer threshold discretization
        // (small committees with tight integer cutoffs can be more fragile
        // than large committees with slack).
        let sizes = [5u32, 20, 50, 100];
        let mut last_messages = 0u64;
        for &n in &sizes {
            let t = (n * 2).div_ceil(3);
            let cfg = DkgConfig::new(n, t, 1, 0.10);
            let cost = DkgCost::estimate(&cfg);
            assert!(
                cost.message_count > last_messages,
                "messages not monotone at n={n}",
            );
            assert_eq!(cost.latency_ticks, 2, "latency independent of n");
            last_messages = cost.message_count;
        }
        // At n=100, message count should be ~two orders of magnitude above n=5.
        let n_100 = DkgCost::estimate(&DkgConfig::new(100, 67, 1, 0.10));
        let n_5 = DkgCost::estimate(&DkgConfig::new(5, 4, 1, 0.10));
        assert!(n_100.message_count > 100 * n_5.message_count);
    }

    #[test]
    fn binomial_sanity() {
        // P(X ≥ 1) under Bin(n, p) = 1 − (1 − p)^n
        assert!((binomial_tail_ge(10, 1, 0.1) - 0.6513215599).abs() < 1e-6);
        // P(X ≥ n) = p^n
        let p: f64 = 0.5;
        let n: u32 = 5;
        let expected = p.powi(n as i32);
        assert!((binomial_tail_ge(n, n, p) - expected).abs() < 1e-9);
    }
}
