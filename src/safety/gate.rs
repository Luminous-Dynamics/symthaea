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
pub fn consciousness_gate(consciousness_level: f32, min_threshold: f32) -> SafetyGateResult {
    if consciousness_level >= min_threshold {
        SafetyGateResult::Proceed
    } else {
        SafetyGateResult::Blocked {
            level: SafetyLevel::Red,
            reason: format!(
                "Consciousness {:.3} below minimum threshold {:.3}",
                consciousness_level, min_threshold
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
}
