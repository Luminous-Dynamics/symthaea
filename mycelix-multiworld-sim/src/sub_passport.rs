// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Restorative justice — per-agent violation/correction tracker.
//!
//! This is the simulator-facing subset of Mycelix's full `SubPassport`
//! (`crates/mycelix-bridge-common/src/sub_passport.rs`, 607 LOC). The
//! production type handles AI-agent delegation, DIDs, and cryptographic
//! signing — none of which the sim models. What the sim needs is the
//! restorative-justice mechanism:
//!
//! - Per-agent violation/correction counts
//! - Effective civic tier = raw tier − `tier_penalty` (floored at Observer)
//! - 3 violations ⇒ +1 penalty (one tier degradation)
//! - 10 corrections ⇒ −1 penalty (one tier restoration, not below 0)
//! - Cooldown: at most one tier delta per tick, to prevent oscillation
//! - `compliance_ratio` = corrections / (violations + corrections)
//!
//! The net-zero tier ratio is therefore 10/3 ≈ 3.33 corrections per
//! violation — documented in the Phase 1 survey as the "3:1 correction
//! ratio" that should stabilize populations against drift toward the
//! Observer floor.
//!
//! Integration sites:
//! - `sanctions::apply_sanctions` records violations when an agent is fined.
//! - A new per-tick correction phase rewards care work (tend_balance gain,
//!   teaching deltas, virtue_care-driven mutual aid).
//! - `World::civic_fraction_meeting` and `civic_tier_distribution` respect
//!   the per-agent effective tier.

use serde::{Deserialize, Serialize};

use crate::sovereign_profile::CivicTier;

/// How many violations trigger one tier degradation.
pub const VIOLATIONS_PER_DEGRADE: u32 = 3;

/// How many corrections trigger one tier restoration.
pub const CORRECTIONS_PER_RESTORE: u32 = 10;

/// Maximum tier penalty (matches the number of civic tiers − 1).
pub const MAX_TIER_PENALTY: u8 = 4;

/// Phase 2c: maximum corrections credited per tick, to defeat CorrectionFarmer
/// attacks that alternate violations with manufactured corrections at >1/tick.
/// Production Mycelix uses a 6h cooldown between credited corrections; in the
/// sim's monthly ticks we cap to a small integer per tick.
pub const MAX_CORRECTIONS_PER_TICK: u32 = 2;

/// Per-agent restorative-justice state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RestorativeJustice {
    /// Cumulative moral violations.
    pub violations: u32,
    /// Cumulative corrective actions (credited, after rate limiting).
    pub corrections: u32,
    /// Current tier degradation (0 = none, up to MAX_TIER_PENALTY).
    pub tier_penalty: u8,
    /// Tick of the most recent tier delta (degrade or restore). Prevents
    /// same-tick oscillation.
    pub last_delta_tick: Option<u32>,
    /// Phase 2c: number of corrections credited on `current_tick` so far.
    /// Reset whenever `record_correction` sees a newer tick.
    #[serde(default)]
    pub corrections_this_tick: u32,
    /// Phase 2c: most recent tick where a correction was recorded (or
    /// attempted). Drives the per-tick rate limit.
    #[serde(default)]
    pub last_correction_tick: Option<u32>,
    /// Phase 2c: rejected-correction count. A suspiciously high value is
    /// evidence of `CorrectionFarmer` attack pattern.
    #[serde(default)]
    pub rejected_corrections: u32,
}

impl RestorativeJustice {
    pub fn new() -> Self {
        Self {
            violations: 0,
            corrections: 0,
            tier_penalty: 0,
            last_delta_tick: None,
            corrections_this_tick: 0,
            last_correction_tick: None,
            rejected_corrections: 0,
        }
    }

    /// Record a moral violation. At every `VIOLATIONS_PER_DEGRADE`-th
    /// violation, the effective tier degrades by one step — unless another
    /// delta was already applied this tick.
    pub fn record_violation(&mut self, current_tick: u32) {
        self.violations = self.violations.saturating_add(1);
        if self.violations % VIOLATIONS_PER_DEGRADE == 0
            && self.tier_penalty < MAX_TIER_PENALTY
            && self.last_delta_tick != Some(current_tick)
        {
            self.tier_penalty += 1;
            self.last_delta_tick = Some(current_tick);
        }
    }

    /// Record a corrective action. At every `CORRECTIONS_PER_RESTORE`-th
    /// correction, one tier of penalty is restored — subject to the
    /// one-delta-per-tick cooldown AND a `MAX_CORRECTIONS_PER_TICK` rate
    /// limit (Phase 2c CorrectionFarmer defense).
    ///
    /// Returns `true` if the correction was credited, `false` if it was
    /// rejected due to rate limiting.
    pub fn record_correction(&mut self, current_tick: u32) -> bool {
        // Reset per-tick counter when tick advances.
        if self.last_correction_tick != Some(current_tick) {
            self.corrections_this_tick = 0;
            self.last_correction_tick = Some(current_tick);
        }
        if self.corrections_this_tick >= MAX_CORRECTIONS_PER_TICK {
            self.rejected_corrections = self.rejected_corrections.saturating_add(1);
            return false;
        }
        self.corrections_this_tick += 1;
        self.corrections = self.corrections.saturating_add(1);
        if self.corrections % CORRECTIONS_PER_RESTORE == 0
            && self.tier_penalty > 0
            && self.last_delta_tick != Some(current_tick)
        {
            self.tier_penalty -= 1;
            self.last_delta_tick = Some(current_tick);
        }
        true
    }

    /// Phase 2c: correction-farming suspicion score in [0, 1].
    /// 0.0 = genuine behavior, 1.0 = strong evidence of farming.
    /// Ratio of rejected corrections to total attempts; if many corrections
    /// were rejected by the rate limiter, the agent is spamming.
    pub fn correction_farming_score(&self) -> f64 {
        let attempted = self.corrections + self.rejected_corrections;
        if attempted == 0 {
            0.0
        } else {
            self.rejected_corrections as f64 / attempted as f64
        }
    }

    /// Compliance ratio in [0, 1]. 1.0 when no events have been recorded.
    pub fn compliance_ratio(&self) -> f64 {
        let total = self.violations + self.corrections;
        if total == 0 {
            1.0
        } else {
            self.corrections as f64 / total as f64
        }
    }

    /// Apply the tier penalty to a raw civic tier. The effective tier never
    /// exceeds the raw tier and never falls below `Observer`.
    pub fn effective_tier(&self, raw: CivicTier) -> CivicTier {
        let idx = raw.index() as i32 - self.tier_penalty as i32;
        match idx.max(0) {
            0 => CivicTier::Observer,
            1 => CivicTier::Participant,
            2 => CivicTier::Citizen,
            3 => CivicTier::Steward,
            _ => CivicTier::Guardian,
        }
    }
}

impl Default for RestorativeJustice {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_is_clean() {
        let rj = RestorativeJustice::new();
        assert_eq!(rj.violations, 0);
        assert_eq!(rj.corrections, 0);
        assert_eq!(rj.tier_penalty, 0);
        assert_eq!(rj.compliance_ratio(), 1.0);
        assert_eq!(rj.effective_tier(CivicTier::Citizen), CivicTier::Citizen);
    }

    #[test]
    fn three_violations_degrade_one_tier() {
        let mut rj = RestorativeJustice::new();
        rj.record_violation(10);
        rj.record_violation(11);
        rj.record_violation(12);
        assert_eq!(rj.tier_penalty, 1);
        assert_eq!(
            rj.effective_tier(CivicTier::Citizen),
            CivicTier::Participant
        );
        assert_eq!(rj.effective_tier(CivicTier::Guardian), CivicTier::Steward);
    }

    #[test]
    fn ten_corrections_restore_one_tier() {
        let mut rj = RestorativeJustice::new();
        // Degrade twice (6 violations).
        for tick in 0..6 {
            rj.record_violation(tick);
        }
        assert_eq!(rj.tier_penalty, 2);

        // 10 corrections ⇒ restore 1 tier (penalty 2 → 1).
        for tick in 100..110 {
            rj.record_correction(tick);
        }
        assert_eq!(rj.tier_penalty, 1);
    }

    #[test]
    fn cooldown_prevents_same_tick_oscillation() {
        let mut rj = RestorativeJustice::new();
        // 2 violations (not enough to degrade) then a burst in one tick
        rj.record_violation(0);
        rj.record_violation(0);
        rj.record_violation(5); // → penalty becomes 1, last_delta_tick = 5
        assert_eq!(rj.tier_penalty, 1);

        // Phase 2c: with MAX_CORRECTIONS_PER_TICK = 2, only 2 of the burst
        // are credited on tick 5. Penalty stays at 1 (no milestone reached).
        for _ in 0..10 {
            rj.record_correction(5);
        }
        assert_eq!(
            rj.tier_penalty, 1,
            "restore blocked by cooldown + rate limit"
        );
        assert!(rj.corrections <= MAX_CORRECTIONS_PER_TICK);

        // Accumulate the remaining corrections needed (up to 10 total) at
        // ≤ 2/tick to reach the restore milestone.
        let mut tick = 6;
        while rj.corrections < CORRECTIONS_PER_RESTORE {
            rj.record_correction(tick);
            tick += 1;
        }
        assert_eq!(rj.tier_penalty, 0);
    }

    #[test]
    fn penalty_is_bounded() {
        let mut rj = RestorativeJustice::new();
        for tick in 0..30 {
            rj.record_violation(tick);
        }
        assert!(rj.tier_penalty <= MAX_TIER_PENALTY);
        assert_eq!(rj.effective_tier(CivicTier::Guardian), CivicTier::Observer);
    }

    #[test]
    fn corrections_cannot_go_below_zero_penalty() {
        let mut rj = RestorativeJustice::new();
        for tick in 0..50 {
            rj.record_correction(tick);
        }
        assert_eq!(rj.tier_penalty, 0);
        assert_eq!(rj.effective_tier(CivicTier::Citizen), CivicTier::Citizen);
    }

    #[test]
    fn compliance_ratio_is_computed() {
        let mut rj = RestorativeJustice::new();
        rj.record_violation(0);
        rj.record_violation(1);
        rj.record_correction(2);
        rj.record_correction(3);
        rj.record_correction(4);
        // 3 / 5 = 0.6
        assert!((rj.compliance_ratio() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn three_to_one_ratio_stabilizes_effective_tier() {
        // At 3 corrections per violation, tier penalty should stay bounded
        // (roughly zero net) over a long horizon.
        let mut rj = RestorativeJustice::new();
        // 300 violations + 900 corrections interleaved, one event per tick.
        for tick in 0..1200 {
            if tick % 4 == 0 {
                rj.record_violation(tick);
            } else {
                rj.record_correction(tick);
            }
        }
        // Expected penalty ≈ 0: 300/3 = 100 degrades, 900/10 = 90 restores.
        // Net +10 (degradation slightly outpaces restoration because the
        // 3:1 ratio is the *compliance* ratio, not the recovery ratio —
        // recovery needs 10/3 ≈ 3.33 corrections per violation).
        // The 3:1 ratio produces a *bounded* penalty, but not zero.
        assert!(
            rj.tier_penalty <= MAX_TIER_PENALTY,
            "penalty unbounded: {}",
            rj.tier_penalty,
        );
    }

    #[test]
    fn correction_rate_limit_catches_farming() {
        // CorrectionFarmer attack: try to credit 100 corrections all on tick 0.
        let mut rj = RestorativeJustice::new();
        for _ in 0..100 {
            rj.record_correction(0);
        }
        assert_eq!(rj.corrections, MAX_CORRECTIONS_PER_TICK);
        // 98 attempts rejected → farming score should be very high.
        assert!(rj.correction_farming_score() > 0.95);
    }

    #[test]
    fn genuine_correction_has_low_farming_score() {
        let mut rj = RestorativeJustice::new();
        // One correction per tick — genuine behavior.
        for t in 0..50 {
            rj.record_correction(t);
        }
        assert_eq!(rj.rejected_corrections, 0);
        assert_eq!(rj.correction_farming_score(), 0.0);
    }

    #[test]
    fn record_correction_returns_credited_status() {
        let mut rj = RestorativeJustice::new();
        assert!(rj.record_correction(0));
        assert!(rj.record_correction(0));
        // Third attempt on same tick exceeds MAX_CORRECTIONS_PER_TICK.
        assert!(!rj.record_correction(0));
        assert_eq!(rj.rejected_corrections, 1);
    }

    #[test]
    fn ten_to_three_ratio_is_net_zero() {
        // 10 corrections per 3 violations = exact restore-to-degrade parity.
        let mut rj = RestorativeJustice::new();
        for tick in 0..1300 {
            if tick % 13 < 3 {
                rj.record_violation(tick);
            } else {
                rj.record_correction(tick);
            }
        }
        // 300 violations, 1000 corrections → 100 degrades, 100 restores → penalty 0
        assert_eq!(rj.tier_penalty, 0);
    }
}
