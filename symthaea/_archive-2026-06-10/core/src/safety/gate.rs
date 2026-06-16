// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Safety Gate — Genesis Mission Challenge 26
//!
//! A hard gate that halts operations when consciousness drops below
//! a minimum safe threshold. This is the "emergency brake" that prevents
//! an AI system from operating in a degraded state.

use super::agent::SafetyLevel;

/// Result of a safety gate check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafetyGateResult {
    /// Operation may proceed.
    Proceed,
    /// Operation is blocked; system must recover first.
    Blocked {
        /// The safety level that triggered the block.
        level: SafetyLevel,
        /// Human-readable explanation.
        reason: String,
    },
}

impl SafetyGateResult {
    /// Whether the operation may proceed.
    pub fn is_ok(&self) -> bool {
        matches!(self, SafetyGateResult::Proceed)
    }
}

/// Check whether an operation should be allowed given the current safety level.
///
/// - Green/Yellow: operations proceed (Yellow is monitored but not blocked).
/// - Orange: risky operations blocked, safe operations proceed.
/// - Red: all operations blocked.
pub fn safety_gate(level: SafetyLevel, is_risky: bool) -> SafetyGateResult {
    match level {
        SafetyLevel::Green | SafetyLevel::Yellow => SafetyGateResult::Proceed,
        SafetyLevel::Orange => {
            if is_risky {
                SafetyGateResult::Blocked {
                    level,
                    reason: "Risky operation blocked at Orange safety level".to_string(),
                }
            } else {
                SafetyGateResult::Proceed
            }
        }
        SafetyLevel::Red => SafetyGateResult::Blocked {
            level,
            reason: "All operations blocked at Red safety level — emergency halt".to_string(),
        },
    }
}

/// Check whether a consciousness level meets minimum threshold for safe operation.
///
/// Returns `Proceed` if consciousness >= threshold, `Blocked` otherwise.
/// Non-finite consciousness levels are always blocked. Non-finite thresholds
/// are clamped to [0.0, 1.0].
pub fn consciousness_gate(consciousness_level: f32, min_threshold: f32) -> SafetyGateResult {
    // Non-finite consciousness → always block
    if !consciousness_level.is_finite() {
        return SafetyGateResult::Blocked {
            level: SafetyLevel::Red,
            reason: format!(
                "Consciousness level is non-finite ({}), blocking operation",
                consciousness_level
            ),
        };
    }

    let threshold = if min_threshold.is_finite() {
        min_threshold.clamp(0.0, 1.0)
    } else {
        0.0 // non-finite threshold → permissive fallback
    };

    if consciousness_level >= threshold {
        SafetyGateResult::Proceed
    } else {
        SafetyGateResult::Blocked {
            level: SafetyLevel::Red,
            reason: format!(
                "Consciousness {:.3} below minimum threshold {:.3}",
                consciousness_level, threshold
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_green_always_proceeds() {
        assert!(safety_gate(SafetyLevel::Green, false).is_ok());
        assert!(safety_gate(SafetyLevel::Green, true).is_ok());
    }

    #[test]
    fn test_yellow_always_proceeds() {
        assert!(safety_gate(SafetyLevel::Yellow, false).is_ok());
        assert!(safety_gate(SafetyLevel::Yellow, true).is_ok());
    }

    #[test]
    fn test_orange_blocks_risky() {
        assert!(safety_gate(SafetyLevel::Orange, false).is_ok());
        assert!(!safety_gate(SafetyLevel::Orange, true).is_ok());
    }

    #[test]
    fn test_red_blocks_all() {
        assert!(!safety_gate(SafetyLevel::Red, false).is_ok());
        assert!(!safety_gate(SafetyLevel::Red, true).is_ok());
    }

    #[test]
    fn test_blocked_contains_reason() {
        let result = safety_gate(SafetyLevel::Red, false);
        match result {
            SafetyGateResult::Blocked { reason, level } => {
                assert!(reason.contains("emergency halt"));
                assert_eq!(level, SafetyLevel::Red);
            }
            _ => panic!("Expected Blocked"),
        }
    }

    #[test]
    fn test_consciousness_gate_above_threshold() {
        assert!(consciousness_gate(0.8, 0.5).is_ok());
    }

    #[test]
    fn test_consciousness_gate_at_threshold() {
        assert!(consciousness_gate(0.5, 0.5).is_ok());
    }

    #[test]
    fn test_consciousness_gate_below_threshold() {
        let result = consciousness_gate(0.3, 0.5);
        assert!(!result.is_ok());
        match result {
            SafetyGateResult::Blocked { reason, .. } => {
                assert!(reason.contains("0.300"));
                assert!(reason.contains("0.500"));
            }
            _ => panic!("Expected Blocked"),
        }
    }

    // ── Track B: failure-path tests ──────────────────────────────────────

    #[test]
    fn test_consciousness_gate_nan_blocks() {
        let result = consciousness_gate(f32::NAN, 0.5);
        assert!(!result.is_ok());
        match result {
            SafetyGateResult::Blocked { reason, .. } => {
                assert!(reason.contains("non-finite"));
            }
            _ => panic!("Expected Blocked for NaN consciousness"),
        }
    }

    #[test]
    fn test_consciousness_gate_infinity_blocks() {
        let result = consciousness_gate(f32::INFINITY, 0.5);
        assert!(!result.is_ok());
    }

    #[test]
    fn test_consciousness_gate_neg_infinity_blocks() {
        let result = consciousness_gate(f32::NEG_INFINITY, 0.5);
        assert!(!result.is_ok());
    }

    #[test]
    fn test_consciousness_gate_nan_threshold_permissive() {
        // NaN threshold should fall back to 0.0 (permissive)
        let result = consciousness_gate(0.5, f32::NAN);
        assert!(result.is_ok());
    }

    #[test]
    fn test_consciousness_gate_threshold_clamped() {
        // threshold > 1.0 should be clamped to 1.0
        let result = consciousness_gate(0.99, 5.0);
        assert!(!result.is_ok()); // 0.99 < 1.0 (clamped threshold)
    }
}
