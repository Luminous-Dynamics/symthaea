// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Safety tier system: Φ-gated motor authority.
//!
//! Maps integrated information (Φ) to motor gain via an NRC-inspired 4-tier
//! system. Higher Φ (more integrated information processing) grants more
//! force authority. This is a formal coupling, not a philosophical claim —
//! it could equally be called "integration-gated motor authority."

/// Φ-gated safety tier for motor/force output.
///
/// Based on NRC (Nuclear Regulatory Commission) 4-tier safety model,
/// mapped to integrated information level (Φ):
///
/// | Tier   | Φ Range   | Motor Gain | Behavior                |
/// |--------|-----------|------------|-------------------------|
/// | Green  | > 0.6     | 100%       | Full authority           |
/// | Yellow | 0.3–0.6   | 60%        | Reduced speed/force      |
/// | Orange | 0.1–0.3   | 30%        | Retreat to safe pose     |
/// | Red    | ≤ 0.1     | 0%         | Emergency stop           |
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SafetyTier {
    Green,
    Yellow,
    Orange,
    Red,
}

impl SafetyTier {
    /// Determine safety tier from Φ value.
    pub fn from_phi(phi: f64) -> Self {
        if phi > 0.6 {
            Self::Green
        } else if phi > 0.3 {
            Self::Yellow
        } else if phi > 0.1 {
            Self::Orange
        } else {
            Self::Red
        }
    }

    /// Motor gain multiplier [0.0, 1.0].
    #[inline]
    pub fn motor_gain(&self) -> f64 {
        match self {
            Self::Green => 1.0,
            Self::Yellow => 0.6,
            Self::Orange => 0.3,
            Self::Red => 0.0,
        }
    }

    /// Whether this tier allows any motor output.
    #[inline]
    pub fn allows_output(&self) -> bool {
        *self != Self::Red
    }
}

/// Minimal 2-part Φ → gain mapping, empirically derived.
///
/// For a given platform the consciousness equation tends to produce Φ
/// values concentrated in a narrow band rather than spanning [0, 1].
/// The 5-variant Monte Carlo study in
/// `symtropy-manipulator-demo/examples/manipulator_benchmark.rs`
/// (see commits `38dc8b1fd9` through `317baad595`, Apr 2026) found
/// that a Φ→gain mapping beats binary ISO/TS 15066 SSM (if at all)
/// only when it satisfies **both**:
///
///   1. A **sprint threshold**: a Φ value at or below the top of the
///      platform's empirical band, above which `gain = 1.0` so the
///      controller commits to full authority in brief confident windows.
///   2. A **crawl-rate floor**: a non-zero gain below the sprint
///      threshold, large enough to keep IK convergence alive. The
///      surgical/manipulator demos use 0.3; adjust for your
///      convergence-tolerance vs controller-gains.
///
/// Intermediate tiers (0.6, 0.3 steps as in `SafetyTier::motor_gain`)
/// proved empirically superfluous in the manipulator study — a
/// two-level `sprint_floor_gain` matched the 4-tier `Recalibrated`
/// variant to three decimal places.
///
/// **`SafetyTier::from_phi` uses hardcoded Φ thresholds (0.6 / 0.3 /
/// 0.1) that are unlikely to match any platform's empirical Φ band.**
/// For a working Φ-gated demo, measure your platform's Φ distribution
/// first (e.g. via the `MANIP_BENCH_PHI_TRACE=1` pattern in the
/// manipulator benchmark), then pass your platform-specific
/// `sprint_phi` and `floor_gain` to this function.
///
/// # Example
///
/// ```
/// use symtropy_consciousness_physics::safety::sprint_floor_gain;
///
/// // Platform's empirical Φ band is [0.099, 0.145].
/// // Pick sprint threshold near the top, floor above the crawl limit.
/// let gain = sprint_floor_gain(0.14, 0.135, 0.3);
/// assert!((gain - 1.0).abs() < 1e-9);  // Φ above sprint → full authority.
///
/// let gain = sprint_floor_gain(0.11, 0.135, 0.3);
/// assert!((gain - 0.3).abs() < 1e-9);  // Φ below sprint → floor.
/// ```
///
/// # Panics
///
/// Does not panic. `floor` outside [0, 1] is accepted but the caller
/// is responsible for choosing a physically sensible value — typically
/// above whatever gain floor the inverse-kinematics loop needs to
/// converge, and below 1.0.
#[inline]
pub fn sprint_floor_gain(phi: f64, sprint_phi: f64, floor: f64) -> f64 {
    if phi > sprint_phi {
        1.0
    } else {
        floor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_thresholds() {
        assert_eq!(SafetyTier::from_phi(0.8), SafetyTier::Green);
        assert_eq!(SafetyTier::from_phi(0.5), SafetyTier::Yellow);
        assert_eq!(SafetyTier::from_phi(0.2), SafetyTier::Orange);
        assert_eq!(SafetyTier::from_phi(0.05), SafetyTier::Red);
        assert_eq!(SafetyTier::from_phi(0.0), SafetyTier::Red);
        assert_eq!(SafetyTier::from_phi(1.0), SafetyTier::Green);
    }

    #[test]
    fn motor_gain_values() {
        assert_eq!(SafetyTier::Green.motor_gain(), 1.0);
        assert_eq!(SafetyTier::Yellow.motor_gain(), 0.6);
        assert_eq!(SafetyTier::Orange.motor_gain(), 0.3);
        assert_eq!(SafetyTier::Red.motor_gain(), 0.0);
    }

    #[test]
    fn red_blocks_output() {
        assert!(SafetyTier::Green.allows_output());
        assert!(SafetyTier::Yellow.allows_output());
        assert!(SafetyTier::Orange.allows_output());
        assert!(!SafetyTier::Red.allows_output());
    }

    #[test]
    fn monotonic_phi_to_gain() {
        // Higher phi should never give lower gain
        let phis = [0.0, 0.05, 0.1, 0.15, 0.3, 0.35, 0.6, 0.65, 1.0];
        for window in phis.windows(2) {
            let gain_low = SafetyTier::from_phi(window[0]).motor_gain();
            let gain_high = SafetyTier::from_phi(window[1]).motor_gain();
            assert!(
                gain_high >= gain_low,
                "phi {} (gain {}) > phi {} (gain {})",
                window[1],
                gain_high,
                window[0],
                gain_low
            );
        }
    }

    // ── sprint_floor_gain (empirically-derived minimal mapping) ────────

    #[test]
    fn sprint_floor_gain_full_above_threshold() {
        // Strictly above the sprint threshold → full authority.
        assert_eq!(sprint_floor_gain(0.14, 0.135, 0.3), 1.0);
        assert_eq!(sprint_floor_gain(1.0, 0.135, 0.3), 1.0);
        assert_eq!(sprint_floor_gain(0.50, 0.13, 0.25), 1.0);
    }

    #[test]
    fn sprint_floor_gain_floor_at_and_below_threshold() {
        // At exactly the sprint threshold, strict inequality keeps us at floor
        // (the safety margin that would let a paranoid reviewer confirm
        // "sprint doesn't fire until Φ STRICTLY exceeds the threshold").
        assert_eq!(sprint_floor_gain(0.135, 0.135, 0.3), 0.3);
        assert_eq!(sprint_floor_gain(0.134, 0.135, 0.3), 0.3);
        assert_eq!(sprint_floor_gain(0.0, 0.135, 0.3), 0.3);
    }

    #[test]
    fn sprint_floor_gain_with_zero_floor_still_sprints() {
        // Floor of 0.0 is allowed (recovers a sharp-cutoff binary mapping).
        assert_eq!(sprint_floor_gain(0.14, 0.135, 0.0), 1.0);
        assert_eq!(sprint_floor_gain(0.10, 0.135, 0.0), 0.0);
    }

    #[test]
    fn sprint_floor_gain_is_two_level_only() {
        // No intermediate values — every output is either floor or 1.0.
        // This is the "middle tiers are superfluous" finding locked in.
        let floor = 0.3;
        for phi_x100 in 0..200 {
            let phi = phi_x100 as f64 / 100.0;
            let g = sprint_floor_gain(phi, 0.135, floor);
            assert!(
                (g - 1.0).abs() < 1e-12 || (g - floor).abs() < 1e-12,
                "sprint_floor_gain produced intermediate value {} at phi={}",
                g,
                phi
            );
        }
    }
}
