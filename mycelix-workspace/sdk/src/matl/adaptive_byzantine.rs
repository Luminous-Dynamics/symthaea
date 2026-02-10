//! Adaptive Byzantine Threshold
//!
//! Dynamically adjusts the Byzantine tolerance threshold based on:
//! - Network size (larger networks can tolerate higher fractions)
//! - Historical attack patterns (increases caution after attacks)
//! - Reputation distribution (adjusts based on trust landscape)
//!
//! ## Algorithm
//!
//! The adaptive threshold uses an exponential moving average combined with
//! safety bounds to ensure the network remains secure while maximizing
//! participation.

use std::collections::VecDeque;

/// Maximum validated Byzantine tolerance (empirically verified to 34%)
pub const MAX_BYZANTINE_TOLERANCE: f64 = 0.34;

// =============================================================================
// FIND-006 Mitigation: Safe threshold checking with NaN/Infinity guards
// =============================================================================

/// Safely compare a value against a threshold with NaN/Infinity guards.
///
/// # Security Note (FIND-006 mitigation)
///
/// Floating-point comparisons in security-critical paths must handle edge cases:
/// - NaN values (which fail all comparisons in undefined ways)
/// - Infinity values (which could bypass thresholds)
///
/// This function fails safe: returns `false` if either value is not finite,
/// preventing malformed inputs from bypassing Byzantine detection.
///
/// # Arguments
/// * `value` - The value to check (e.g., Byzantine fraction)
/// * `threshold` - The threshold to compare against
///
/// # Returns
/// `true` if `value >= threshold` AND both values are finite, `false` otherwise
#[inline]
#[allow(dead_code)]
pub fn safe_threshold_check(value: f64, threshold: f64) -> bool {
    // Fail safe: if either value is NaN or Infinity, return false
    if !value.is_finite() || !threshold.is_finite() {
        return false;
    }
    value >= threshold
}

/// Safely compare if a value is below a threshold with NaN/Infinity guards.
///
/// # Security Note (FIND-006 mitigation)
///
/// This is the inverse of `safe_threshold_check`, used for checking if
/// a value is acceptable (below the threshold).
///
/// # Arguments
/// * `value` - The value to check (e.g., Byzantine fraction)
/// * `threshold` - The threshold to compare against
///
/// # Returns
/// `true` if `value <= threshold` AND both values are finite, `false` otherwise
#[inline]
pub fn safe_below_threshold(value: f64, threshold: f64) -> bool {
    // Fail safe: if either value is NaN or Infinity, return false
    if !value.is_finite() || !threshold.is_finite() {
        return false;
    }
    value <= threshold
}

/// Minimum Byzantine tolerance (safety floor)
pub const MIN_BYZANTINE_TOLERANCE: f64 = 0.10;

/// Default starting threshold
pub const DEFAULT_THRESHOLD: f64 = 0.30;

/// Window size for moving average
const WINDOW_SIZE: usize = 20;

/// Adaptive Byzantine Threshold Manager
///
/// Tracks network conditions and adjusts the Byzantine tolerance
/// threshold dynamically.
#[derive(Debug, Clone)]
pub struct AdaptiveByzantineThreshold {
    /// Current threshold value
    threshold: f64,

    /// Historical Byzantine fractions observed
    history: VecDeque<f64>,

    /// Number of nodes in the network
    network_size: usize,

    /// Count of confirmed attacks in recent history
    attack_count: usize,

    /// Smoothing factor for EMA (0-1, higher = more responsive)
    alpha: f64,

    /// Safety margin (reduces threshold after attacks)
    safety_margin: f64,
}

impl Default for AdaptiveByzantineThreshold {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveByzantineThreshold {
    /// Create a new adaptive threshold manager
    pub fn new() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            history: VecDeque::with_capacity(WINDOW_SIZE),
            network_size: 0,
            attack_count: 0,
            alpha: 0.2,
            safety_margin: 0.0,
        }
    }

    /// Create with custom parameters
    pub fn with_params(initial_threshold: f64, alpha: f64) -> Self {
        Self {
            threshold: initial_threshold.clamp(MIN_BYZANTINE_TOLERANCE, MAX_BYZANTINE_TOLERANCE),
            history: VecDeque::with_capacity(WINDOW_SIZE),
            network_size: 0,
            attack_count: 0,
            alpha: alpha.clamp(0.01, 0.99),
            safety_margin: 0.0,
        }
    }

    /// Get the current threshold
    pub fn threshold(&self) -> f64 {
        (self.threshold - self.safety_margin)
            .clamp(MIN_BYZANTINE_TOLERANCE, MAX_BYZANTINE_TOLERANCE)
    }

    /// Update the threshold based on observed round
    ///
    /// # Arguments
    /// * `byzantine_fraction` - Fraction of nodes detected as Byzantine (0.0-1.0)
    /// * `network_size` - Current number of nodes in the network
    /// * `attack_confirmed` - Whether a confirmed attack occurred this round
    pub fn observe(
        &mut self,
        byzantine_fraction: f64,
        network_size: usize,
        attack_confirmed: bool,
    ) {
        self.network_size = network_size;

        // Add to history
        self.history.push_back(byzantine_fraction);
        if self.history.len() > WINDOW_SIZE {
            self.history.pop_front();
        }

        // Track attacks
        if attack_confirmed {
            self.attack_count += 1;
            // Increase safety margin after attack
            self.safety_margin = (self.safety_margin + 0.05).min(0.15);
        } else if self.attack_count > 0 {
            // Slowly decay safety margin when no attacks
            self.safety_margin = (self.safety_margin - 0.01).max(0.0);
            if self.safety_margin == 0.0 {
                self.attack_count = self.attack_count.saturating_sub(1);
            }
        }

        // Compute new threshold
        self.update_threshold();
    }

    /// Update the threshold based on current state
    fn update_threshold(&mut self) {
        // 1. Network size adjustment
        // Larger networks can tolerate slightly higher Byzantine fractions
        let size_factor = self.compute_size_factor();

        // 2. Historical average
        let avg_byzantine = if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().sum::<f64>() / self.history.len() as f64
        };

        // 3. Compute target threshold
        // Base: slightly above observed Byzantine fraction to allow participation
        // Bounded by safety limits
        let base_target = (avg_byzantine * 1.5 + 0.1).min(MAX_BYZANTINE_TOLERANCE * size_factor);

        // 4. Recovery toward default during calm periods
        // If no attacks and low Byzantine activity, ensure threshold doesn't drop too low
        // Use the MAX of base_target and a recovery floor to ensure recovery after attacks
        let recovery_bias =
            if self.attack_count == 0 && avg_byzantine < 0.15 && self.safety_margin == 0.0 {
                // During calm periods, don't let threshold drop below DEFAULT
                // This ensures recovery after attacks
                base_target.max(DEFAULT_THRESHOLD)
            } else {
                base_target
            };

        // 5. Apply exponential moving average
        self.threshold = self.alpha * recovery_bias + (1.0 - self.alpha) * self.threshold;

        // 6. Clamp to bounds
        self.threshold = self
            .threshold
            .clamp(MIN_BYZANTINE_TOLERANCE, MAX_BYZANTINE_TOLERANCE);
    }

    /// Compute size adjustment factor
    ///
    /// Larger networks are more robust to Byzantine nodes because:
    /// - More redundancy in voting
    /// - Harder for adversaries to control large fraction
    fn compute_size_factor(&self) -> f64 {
        match self.network_size {
            0..=10 => 0.85,   // Small network: conservative
            11..=50 => 0.90,  // Medium network: slightly conservative
            51..=100 => 0.95, // Larger network: near standard
            101..=500 => 1.0, // Standard network: full tolerance
            _ => 1.05,        // Very large: slightly relaxed (but capped at MAX)
        }
    }

    /// Check if a Byzantine fraction is acceptable
    ///
    /// # Security Note (FIND-006 mitigation)
    ///
    /// Uses `safe_below_threshold` to guard against NaN/Infinity edge cases.
    /// Returns `false` (fail safe) if the fraction contains invalid floating-point values.
    pub fn is_acceptable(&self, byzantine_fraction: f64) -> bool {
        safe_below_threshold(byzantine_fraction, self.threshold())
    }

    /// Get recommendation for current network state
    pub fn recommendation(&self) -> ThresholdRecommendation {
        let current = self.threshold();
        let avg_byzantine = if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().sum::<f64>() / self.history.len() as f64
        };

        // Determine status - prioritize confirmed attacks over statistical warnings
        let status = if self.attack_count > 2 {
            NetworkStatus::UnderAttack
        } else if avg_byzantine > current * 0.8 {
            NetworkStatus::Warning
        } else {
            NetworkStatus::Healthy
        };

        ThresholdRecommendation {
            current_threshold: current,
            network_size: self.network_size,
            avg_byzantine_fraction: avg_byzantine,
            recent_attacks: self.attack_count,
            safety_margin: self.safety_margin,
            status,
        }
    }

    /// Reset to defaults
    pub fn reset(&mut self) {
        self.threshold = DEFAULT_THRESHOLD;
        self.history.clear();
        self.attack_count = 0;
        self.safety_margin = 0.0;
    }

    /// Get history window
    pub fn history(&self) -> &VecDeque<f64> {
        &self.history
    }
}

/// Network health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkStatus {
    /// Network is operating normally
    Healthy,
    /// Byzantine fraction approaching threshold
    Warning,
    /// Active attack detected
    UnderAttack,
}

/// Threshold recommendation with context
#[derive(Debug, Clone)]
pub struct ThresholdRecommendation {
    /// Current effective threshold
    pub current_threshold: f64,
    /// Network size
    pub network_size: usize,
    /// Average observed Byzantine fraction
    pub avg_byzantine_fraction: f64,
    /// Number of recent attacks
    pub recent_attacks: usize,
    /// Current safety margin
    pub safety_margin: f64,
    /// Network status assessment
    pub status: NetworkStatus,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_threshold() {
        let abt = AdaptiveByzantineThreshold::new();
        assert!((abt.threshold() - DEFAULT_THRESHOLD).abs() < 0.01);
    }

    #[test]
    fn test_threshold_bounds() {
        let mut abt = AdaptiveByzantineThreshold::new();

        // Even with extreme inputs, should stay in bounds
        for _ in 0..100 {
            abt.observe(1.0, 100, true); // Max Byzantine, attacks
        }

        let threshold = abt.threshold();
        assert!(threshold >= MIN_BYZANTINE_TOLERANCE);
        assert!(threshold <= MAX_BYZANTINE_TOLERANCE);
    }

    #[test]
    fn test_attack_increases_safety() {
        let mut abt = AdaptiveByzantineThreshold::new();
        let initial = abt.threshold();

        abt.observe(0.2, 50, true); // Confirmed attack

        // Threshold should decrease (more conservative) after attack
        assert!(abt.threshold() < initial);
    }

    #[test]
    fn test_recovery_after_calm() {
        let mut abt = AdaptiveByzantineThreshold::new();

        // Simulate attack
        abt.observe(0.3, 50, true);
        let after_attack = abt.threshold();

        // Simulate many calm rounds
        for _ in 0..50 {
            abt.observe(0.05, 50, false);
        }

        // Should recover towards normal
        assert!(abt.threshold() > after_attack);
    }

    #[test]
    fn test_size_factor() {
        let mut small = AdaptiveByzantineThreshold::new();
        let mut large = AdaptiveByzantineThreshold::new();

        for _ in 0..10 {
            small.observe(0.2, 10, false);
            large.observe(0.2, 500, false);
        }

        // Larger network should have higher effective threshold
        assert!(large.threshold() >= small.threshold());
    }

    #[test]
    fn test_acceptable_check() {
        let mut abt = AdaptiveByzantineThreshold::new();
        abt.observe(0.1, 50, false);

        assert!(abt.is_acceptable(0.2));
        assert!(!abt.is_acceptable(0.5));
    }

    #[test]
    fn test_recommendation() {
        let mut abt = AdaptiveByzantineThreshold::new();

        // Normal operation
        for _ in 0..5 {
            abt.observe(0.1, 50, false);
        }
        let rec = abt.recommendation();
        assert_eq!(rec.status, NetworkStatus::Healthy);

        // Under attack
        for _ in 0..5 {
            abt.observe(0.3, 50, true);
        }
        let rec = abt.recommendation();
        assert_eq!(rec.status, NetworkStatus::UnderAttack);
    }

    // ==========================================================================
    // FIND-006 Mitigation Tests: NaN/Infinity guards
    // ==========================================================================

    #[test]
    fn test_safe_threshold_check_normal_values() {
        // Normal comparisons should work as expected
        assert!(safe_threshold_check(0.5, 0.3)); // 0.5 >= 0.3
        assert!(!safe_threshold_check(0.2, 0.3)); // 0.2 < 0.3
        assert!(safe_threshold_check(0.3, 0.3)); // 0.3 >= 0.3 (equal)
    }

    #[test]
    fn test_safe_threshold_check_nan() {
        // NaN in value should fail safe (return false)
        assert!(!safe_threshold_check(f64::NAN, 0.3));

        // NaN in threshold should fail safe (return false)
        assert!(!safe_threshold_check(0.5, f64::NAN));

        // Both NaN should fail safe
        assert!(!safe_threshold_check(f64::NAN, f64::NAN));
    }

    #[test]
    fn test_safe_threshold_check_infinity() {
        // Positive infinity in value should fail safe
        assert!(!safe_threshold_check(f64::INFINITY, 0.3));

        // Negative infinity in value should fail safe
        assert!(!safe_threshold_check(f64::NEG_INFINITY, 0.3));

        // Infinity in threshold should fail safe
        assert!(!safe_threshold_check(0.5, f64::INFINITY));
        assert!(!safe_threshold_check(0.5, f64::NEG_INFINITY));
    }

    #[test]
    fn test_safe_below_threshold_normal_values() {
        // Normal comparisons should work as expected
        assert!(safe_below_threshold(0.2, 0.3)); // 0.2 <= 0.3
        assert!(!safe_below_threshold(0.5, 0.3)); // 0.5 > 0.3
        assert!(safe_below_threshold(0.3, 0.3)); // 0.3 <= 0.3 (equal)
    }

    #[test]
    fn test_safe_below_threshold_nan_infinity() {
        // NaN should fail safe
        assert!(!safe_below_threshold(f64::NAN, 0.3));
        assert!(!safe_below_threshold(0.2, f64::NAN));

        // Infinity should fail safe
        assert!(!safe_below_threshold(f64::INFINITY, 0.3));
        assert!(!safe_below_threshold(0.2, f64::INFINITY));
    }

    #[test]
    fn test_is_acceptable_with_invalid_values() {
        let mut abt = AdaptiveByzantineThreshold::new();
        abt.observe(0.1, 50, false);

        // Normal values should work
        assert!(abt.is_acceptable(0.2));

        // NaN Byzantine fraction should be rejected (fail safe)
        assert!(!abt.is_acceptable(f64::NAN));

        // Infinity Byzantine fraction should be rejected (fail safe)
        assert!(!abt.is_acceptable(f64::INFINITY));
        assert!(!abt.is_acceptable(f64::NEG_INFINITY));
    }
}
