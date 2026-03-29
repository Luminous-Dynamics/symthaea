//! Safety tier system: consciousness-gated motor authority.
//!
//! Mirrors symthaea's MotorSafetyLevel (NRC-inspired 4-tier system).
//! Re-implemented here to avoid pulling in the full symthaea crate.

/// Consciousness-gated safety tier for motor/force output.
///
/// Based on NRC (Nuclear Regulatory Commission) 4-tier safety model,
/// mapped to consciousness level (Φ):
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
                window[1], gain_high, window[0], gain_low
            );
        }
    }
}
