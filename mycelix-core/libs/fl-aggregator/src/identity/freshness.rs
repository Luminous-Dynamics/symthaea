//! Factor Freshness & Decay
//!
//! Identity factors lose effective strength over time without re-verification.
//! This prevents stale proofs from maintaining high assurance levels indefinitely.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tracks freshness state for each identity factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorFreshness {
    /// Factor ID this freshness tracks
    pub factor_id: String,

    /// Factor type name
    pub factor_type: String,

    /// When the factor was last verified/used
    pub last_verified: DateTime<Utc>,

    /// Factor-specific decay rate (lambda for exponential decay)
    pub decay_rate: f32,

    /// Grace period before decay begins
    pub grace_period: Duration,

    /// When re-verification becomes mandatory
    pub reverification_required: Duration,
}

impl FactorFreshness {
    /// Create new freshness tracker
    pub fn new(
        factor_id: &str,
        factor_type: &str,
        last_verified: DateTime<Utc>,
        config: &FactorDecayConfig,
    ) -> Self {
        Self {
            factor_id: factor_id.to_string(),
            factor_type: factor_type.to_string(),
            last_verified,
            decay_rate: config.decay_rate,
            grace_period: config.grace_period,
            reverification_required: config.reverification_required,
        }
    }

    /// Calculate current effective strength based on time elapsed
    pub fn current_strength(&self, now: DateTime<Utc>) -> f32 {
        let elapsed = now.signed_duration_since(self.last_verified);

        // Within grace period: full strength
        if elapsed <= self.grace_period {
            return 1.0;
        }

        // Exponential decay after grace period
        let decay_time = elapsed - self.grace_period;
        let decay_seconds = decay_time.num_seconds() as f32;

        // Exponential decay: strength = e^(-lambda * t)
        let decay_factor = (-self.decay_rate * decay_seconds / 86400.0).exp(); // Per day

        decay_factor.clamp(0.0, 1.0)
    }

    /// Check if factor requires re-verification
    pub fn needs_reverification(&self, now: DateTime<Utc>) -> bool {
        let elapsed = now.signed_duration_since(self.last_verified);
        self.current_strength(now) < 0.5 || elapsed > self.reverification_required
    }

    /// Get time until re-verification is required
    pub fn time_until_reverification(&self, now: DateTime<Utc>) -> Option<Duration> {
        let elapsed = now.signed_duration_since(self.last_verified);
        if elapsed > self.reverification_required {
            None
        } else {
            Some(self.reverification_required - elapsed)
        }
    }

    /// Mark as re-verified
    pub fn mark_verified(&mut self, now: DateTime<Utc>) {
        self.last_verified = now;
    }

    /// Get freshness status for display
    pub fn status(&self, now: DateTime<Utc>) -> FreshnessStatus {
        let strength = self.current_strength(now);

        if strength >= 0.9 {
            FreshnessStatus::Fresh
        } else if strength >= 0.7 {
            FreshnessStatus::Good
        } else if strength >= 0.5 {
            FreshnessStatus::Stale
        } else if strength >= 0.3 {
            FreshnessStatus::Warning
        } else {
            FreshnessStatus::Expired
        }
    }
}

/// Freshness status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessStatus {
    /// Fully fresh, recently verified
    Fresh,
    /// Good standing, no action needed
    Good,
    /// Getting stale, consider re-verification
    Stale,
    /// Warning, re-verification needed soon
    Warning,
    /// Expired, must re-verify
    Expired,
}

impl FreshnessStatus {
    /// Get display message
    pub fn message(&self) -> &'static str {
        match self {
            FreshnessStatus::Fresh => "Recently verified",
            FreshnessStatus::Good => "Verification current",
            FreshnessStatus::Stale => "Re-verification recommended",
            FreshnessStatus::Warning => "Re-verification required soon",
            FreshnessStatus::Expired => "Must re-verify",
        }
    }
}

/// Decay configuration per factor type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorDecayConfig {
    /// Factor type name
    pub factor_type: String,

    /// Starting strength when fresh
    pub base_strength: f32,

    /// Grace period before decay begins
    pub grace_period: Duration,

    /// Decay rate (lambda for exponential decay)
    pub decay_rate: f32,

    /// Minimum strength floor
    pub minimum_strength: f32,

    /// When re-verification becomes required
    pub reverification_required: Duration,
}

impl FactorDecayConfig {
    /// Get default config for a factor type
    pub fn for_factor_type(factor_type: &str) -> Self {
        match factor_type {
            "CryptoKey" => Self {
                factor_type: "CryptoKey".to_string(),
                base_strength: 1.0,
                grace_period: Duration::days(90),
                decay_rate: 0.002, // Half-life ~365 days
                minimum_strength: 0.3,
                reverification_required: Duration::days(365),
            },
            "HardwareKey" => Self {
                factor_type: "HardwareKey".to_string(),
                base_strength: 1.0,
                grace_period: Duration::days(180),
                decay_rate: 0.001, // Half-life ~730 days
                minimum_strength: 0.4,
                reverification_required: Duration::days(730),
            },
            "Biometric" => Self {
                factor_type: "Biometric".to_string(),
                base_strength: 1.0,
                grace_period: Duration::days(30),
                decay_rate: 0.008, // Half-life ~90 days
                minimum_strength: 0.2,
                reverification_required: Duration::days(90),
            },
            "SocialRecovery" => Self {
                factor_type: "SocialRecovery".to_string(),
                base_strength: 1.0,
                grace_period: Duration::days(60),
                decay_rate: 0.004, // Half-life ~180 days
                minimum_strength: 0.3,
                reverification_required: Duration::days(180),
            },
            "GitcoinPassport" => Self {
                factor_type: "GitcoinPassport".to_string(),
                base_strength: 1.0,
                grace_period: Duration::days(30),
                decay_rate: 0.008, // Half-life ~90 days
                minimum_strength: 0.2,
                reverification_required: Duration::days(90),
            },
            "RecoveryPhrase" => Self {
                factor_type: "RecoveryPhrase".to_string(),
                base_strength: 1.0,
                grace_period: Duration::days(365),
                decay_rate: 0.0, // Never decays (user has it or not)
                minimum_strength: 0.5,
                reverification_required: Duration::days(365 * 5), // 5 years
            },
            _ => Self {
                factor_type: factor_type.to_string(),
                base_strength: 1.0,
                grace_period: Duration::days(90),
                decay_rate: 0.003,
                minimum_strength: 0.2,
                reverification_required: Duration::days(180),
            },
        }
    }
}

/// Manager for tracking factor freshness across an identity
pub struct FreshnessManager {
    /// Freshness states by factor ID
    freshness_states: HashMap<String, FactorFreshness>,

    /// Decay configs by factor type
    decay_configs: HashMap<String, FactorDecayConfig>,
}

impl Default for FreshnessManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FreshnessManager {
    /// Create a new manager with default configs
    pub fn new() -> Self {
        let mut decay_configs = HashMap::new();

        // Register default configs
        for factor_type in &[
            "CryptoKey",
            "HardwareKey",
            "Biometric",
            "SocialRecovery",
            "GitcoinPassport",
            "RecoveryPhrase",
        ] {
            let config = FactorDecayConfig::for_factor_type(factor_type);
            decay_configs.insert(factor_type.to_string(), config);
        }

        Self {
            freshness_states: HashMap::new(),
            decay_configs,
        }
    }

    /// Register a factor for freshness tracking
    pub fn register_factor(
        &mut self,
        factor_id: &str,
        factor_type: &str,
        last_verified: DateTime<Utc>,
    ) {
        let config = self
            .decay_configs
            .get(factor_type)
            .cloned()
            .unwrap_or_else(|| FactorDecayConfig::for_factor_type(factor_type));

        let freshness = FactorFreshness::new(factor_id, factor_type, last_verified, &config);
        self.freshness_states
            .insert(factor_id.to_string(), freshness);
    }

    /// Get freshness for a factor
    pub fn get_freshness(&self, factor_id: &str) -> Option<&FactorFreshness> {
        self.freshness_states.get(factor_id)
    }

    /// Get current strength for a factor
    pub fn get_strength(&self, factor_id: &str, now: DateTime<Utc>) -> f32 {
        self.freshness_states
            .get(factor_id)
            .map(|f| f.current_strength(now))
            .unwrap_or(0.0)
    }

    /// Get all factors needing re-verification
    pub fn factors_needing_reverification(&self, now: DateTime<Utc>) -> Vec<&FactorFreshness> {
        self.freshness_states
            .values()
            .filter(|f| f.needs_reverification(now))
            .collect()
    }

    /// Mark a factor as verified
    pub fn mark_verified(&mut self, factor_id: &str, now: DateTime<Utc>) {
        if let Some(freshness) = self.freshness_states.get_mut(factor_id) {
            freshness.mark_verified(now);
        }
    }

    /// Calculate effective assurance considering freshness decay
    pub fn calculate_effective_strength(
        &self,
        factor_contributions: &[(String, f32)],
        now: DateTime<Utc>,
    ) -> f32 {
        factor_contributions
            .iter()
            .map(|(factor_id, base_contribution)| {
                let freshness_multiplier = self.get_strength(factor_id, now);
                base_contribution * freshness_multiplier
            })
            .sum()
    }

    /// Get freshness summary for display
    pub fn get_summary(&self, now: DateTime<Utc>) -> FreshnessSummary {
        let total_factors = self.freshness_states.len();
        let fresh_count = self
            .freshness_states
            .values()
            .filter(|f| f.status(now) == FreshnessStatus::Fresh)
            .count();
        let stale_count = self
            .freshness_states
            .values()
            .filter(|f| matches!(f.status(now), FreshnessStatus::Stale | FreshnessStatus::Warning))
            .count();
        let expired_count = self
            .freshness_states
            .values()
            .filter(|f| f.status(now) == FreshnessStatus::Expired)
            .count();

        let average_strength: f32 = if total_factors > 0 {
            self.freshness_states
                .values()
                .map(|f| f.current_strength(now))
                .sum::<f32>()
                / total_factors as f32
        } else {
            0.0
        };

        FreshnessSummary {
            total_factors,
            fresh_count,
            stale_count,
            expired_count,
            average_strength,
        }
    }
}

/// Summary of freshness across all factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreshnessSummary {
    pub total_factors: usize,
    pub fresh_count: usize,
    pub stale_count: usize,
    pub expired_count: usize,
    pub average_strength: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freshness_within_grace_period() {
        let config = FactorDecayConfig::for_factor_type("CryptoKey");
        let now = Utc::now();
        let freshness = FactorFreshness::new("test", "CryptoKey", now, &config);

        // Within grace period should be full strength
        assert_eq!(freshness.current_strength(now), 1.0);

        // 30 days in (still within 90 day grace)
        let future = now + Duration::days(30);
        assert_eq!(freshness.current_strength(future), 1.0);
    }

    #[test]
    fn test_freshness_decay_after_grace() {
        let config = FactorDecayConfig::for_factor_type("Biometric");
        let now = Utc::now();
        let freshness = FactorFreshness::new("test", "Biometric", now, &config);

        // After grace period, should start decaying
        let future = now + Duration::days(60); // 30 day grace + 30 days decay
        let strength = freshness.current_strength(future);

        assert!(strength < 1.0);
        assert!(strength > 0.0);
    }

    #[test]
    fn test_freshness_status() {
        let config = FactorDecayConfig::for_factor_type("GitcoinPassport");
        let now = Utc::now();
        let freshness = FactorFreshness::new("test", "GitcoinPassport", now, &config);

        assert_eq!(freshness.status(now), FreshnessStatus::Fresh);

        // Way in the future
        let far_future = now + Duration::days(365);
        let status = freshness.status(far_future);
        assert!(matches!(
            status,
            FreshnessStatus::Warning | FreshnessStatus::Expired
        ));
    }

    #[test]
    fn test_needs_reverification() {
        let config = FactorDecayConfig::for_factor_type("Biometric");
        let now = Utc::now();
        let freshness = FactorFreshness::new("test", "Biometric", now, &config);

        // Shouldn't need reverification now
        assert!(!freshness.needs_reverification(now));

        // Should need reverification after required period
        let future = now + Duration::days(100); // > 90 day requirement
        assert!(freshness.needs_reverification(future));
    }

    #[test]
    fn test_freshness_manager() {
        let mut manager = FreshnessManager::new();
        let now = Utc::now();

        manager.register_factor("crypto-1", "CryptoKey", now);
        manager.register_factor("gitcoin-1", "GitcoinPassport", now - Duration::days(60));

        // Crypto should be fresh
        assert!(manager.get_strength("crypto-1", now) > 0.9);

        // Gitcoin registered 60 days ago should be decaying
        let gitcoin_strength = manager.get_strength("gitcoin-1", now);
        assert!(gitcoin_strength < 1.0);

        // Summary
        let summary = manager.get_summary(now);
        assert_eq!(summary.total_factors, 2);
    }

    #[test]
    fn test_effective_strength_calculation() {
        let mut manager = FreshnessManager::new();
        let now = Utc::now();

        manager.register_factor("crypto-1", "CryptoKey", now);
        manager.register_factor("gitcoin-1", "GitcoinPassport", now - Duration::days(120));

        let contributions = vec![
            ("crypto-1".to_string(), 0.5),
            ("gitcoin-1".to_string(), 0.3),
        ];

        let effective = manager.calculate_effective_strength(&contributions, now);

        // Crypto is fresh (1.0) * 0.5 = 0.5
        // Gitcoin is decayed * 0.3
        assert!(effective < 0.8); // Less than if both were fresh
        assert!(effective > 0.4); // But crypto is still contributing fully
    }

    #[test]
    fn test_recovery_phrase_no_decay() {
        let config = FactorDecayConfig::for_factor_type("RecoveryPhrase");
        assert_eq!(config.decay_rate, 0.0);

        let now = Utc::now();
        let freshness = FactorFreshness::new("phrase-1", "RecoveryPhrase", now, &config);

        // Even years later, should still be within grace
        let far_future = now + Duration::days(300);
        assert_eq!(freshness.current_strength(far_future), 1.0);
    }
}
