// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Metabolic Oracle
//!
//! Implementation of MIP-E-002 Article VII: Autopoietic Self-Regulation
//!
//! The Metabolic Oracle automatically adjusts network economic parameters
//! based on the Vitality Index while respecting constitutional bounds.

use serde::{Deserialize, Serialize};

/// Policy bounds preventing runaway self-modification
/// These are constitutional constraints that cannot be modified by the oracle
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PolicyBounds {
    /// Minimum fee rate (0.1% floor)
    pub fee_rate_min: f64,
    /// Maximum fee rate (3% ceiling)
    pub fee_rate_max: f64,
    /// Minimum decay rate (1% annual floor)
    pub decay_rate_min: f64,
    /// Maximum decay rate (5% annual ceiling)
    pub decay_rate_max: f64,
    /// Minimum SPORE allocation (5/month)
    pub spore_allocation_min: u64,
    /// Maximum SPORE allocation (20/month)
    pub spore_allocation_max: u64,
    /// Emergency reserve minimum (5%)
    pub emergency_reserve_min: f64,
}

impl Default for PolicyBounds {
    fn default() -> Self {
        Self {
            fee_rate_min: 0.001,
            fee_rate_max: 0.03,
            decay_rate_min: 0.01,
            decay_rate_max: 0.05,
            spore_allocation_min: 5,
            spore_allocation_max: 20,
            emergency_reserve_min: 0.05,
        }
    }
}

/// Network vitality measurement components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitalityComponents {
    /// Active SAP / Total SAP × velocity_multiplier
    pub circulation: f64,
    /// Average peer connections / max theoretical
    pub relationship: f64,
    /// HEARTH utilization + CGC flow
    pub commons: f64,
    /// Node count × geographic distribution / target
    pub resilience: f64,
}

impl VitalityComponents {
    /// Calculate composite Vitality Index
    pub fn calculate_vitality(&self) -> f64 {
        const CIRCULATION_WEIGHT: f64 = 0.40;
        const RELATIONSHIP_WEIGHT: f64 = 0.30;
        const COMMONS_WEIGHT: f64 = 0.20;
        const RESILIENCE_WEIGHT: f64 = 0.10;

        let vitality = self.circulation * CIRCULATION_WEIGHT
            + self.relationship * RELATIONSHIP_WEIGHT
            + self.commons * COMMONS_WEIGHT
            + self.resilience * RESILIENCE_WEIGHT;

        (vitality * 100.0).clamp(0.0, 100.0)
    }
}

/// Vitality Index result with state classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitalityIndex {
    /// Raw vitality score (0-100)
    pub score: f64,
    /// Current metabolic state
    pub state: MetabolicState,
    /// Component breakdown
    pub components: VitalityComponents,
    /// Trend direction
    pub trend: VitalityTrend,
    /// Measurement timestamp
    pub timestamp: u64,
}

/// Metabolic state classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetabolicState {
    /// Score 70-100: Network thriving, increase rewards
    Thriving,
    /// Score 40-70: Normal operation
    Healthy,
    /// Score 20-40: Auto-healing activates
    Stressed,
    /// Score 10-20: Emergency response
    Critical,
    /// Score 0-10: Circuit breaker
    Failing,
}

impl MetabolicState {
    /// Determine state from vitality score
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s >= 70.0 => MetabolicState::Thriving,
            s if s >= 40.0 => MetabolicState::Healthy,
            s if s >= 20.0 => MetabolicState::Stressed,
            s if s >= 10.0 => MetabolicState::Critical,
            _ => MetabolicState::Failing,
        }
    }

    /// Check if automatic intervention is needed
    pub fn requires_intervention(&self) -> bool {
        matches!(
            self,
            MetabolicState::Stressed | MetabolicState::Critical | MetabolicState::Failing
        )
    }

    /// Check if circuit breaker should activate
    pub fn circuit_breaker_active(&self) -> bool {
        matches!(self, MetabolicState::Failing)
    }
}

/// Vitality trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VitalityTrend {
    /// Improving over 24h window
    Improving,
    /// Stable (±5%)
    Stable,
    /// Declining over 24h window
    Declining,
    /// Rapid decline (>10% in 24h)
    RapidDecline,
}

/// Policy adjustment recommendation from oracle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyAdjustment {
    /// Fee rate adjustment factor (1.0 = no change)
    pub fee_rate_factor: f64,
    /// SPORE allocation adjustment
    pub spore_adjustment: i32,
    /// Decay rate adjustment factor
    pub decay_rate_factor: f64,
    /// Velocity incentive multiplier
    pub velocity_incentive: f64,
    /// Emergency liquidity release (if critical)
    pub emergency_release: Option<u64>,
    /// Reason for adjustment
    pub reason: String,
    /// Requires human approval (for large changes)
    pub requires_approval: bool,
}

/// The Metabolic Oracle for autopoietic parameter adjustment
#[derive(Debug, Clone)]
pub struct MetabolicOracle {
    /// Constitutional policy bounds
    pub bounds: PolicyBounds,
    /// Current network parameters
    pub current_params: NetworkParameters,
    /// Historical vitality readings (24h rolling)
    pub vitality_history: Vec<VitalityIndex>,
    /// Adjustment history for audit
    pub adjustment_history: Vec<PolicyAdjustment>,
}

/// Current network economic parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkParameters {
    /// Current base fee rate
    pub fee_rate: f64,
    /// Current decay rate (annual)
    pub decay_rate: f64,
    /// Current SPORE allocation per month
    pub spore_allocation: u64,
    /// Velocity incentive multiplier
    pub velocity_incentive: f64,
    /// Emergency reserve ratio
    pub emergency_reserve: f64,
}

impl Default for NetworkParameters {
    fn default() -> Self {
        Self {
            fee_rate: 0.0015, // 0.15%
            decay_rate: 0.02, // 2% annual
            spore_allocation: 10,
            velocity_incentive: 1.0,
            emergency_reserve: 0.10,
        }
    }
}

impl MetabolicOracle {
    /// Create new oracle with default bounds
    pub fn new() -> Self {
        Self {
            bounds: PolicyBounds::default(),
            current_params: NetworkParameters::default(),
            vitality_history: Vec::new(),
            adjustment_history: Vec::new(),
        }
    }

    /// Record new vitality measurement
    pub fn record_vitality(&mut self, components: VitalityComponents, timestamp: u64) {
        let score = components.calculate_vitality();
        let state = MetabolicState::from_score(score);

        let trend = self.calculate_trend(score);

        let vitality = VitalityIndex {
            score,
            state,
            components,
            trend,
            timestamp,
        };

        self.vitality_history.push(vitality);

        // Keep 24h of history (assuming hourly measurements)
        if self.vitality_history.len() > 24 {
            self.vitality_history.remove(0);
        }
    }

    /// Calculate trend from recent history
    fn calculate_trend(&self, current: f64) -> VitalityTrend {
        if self.vitality_history.is_empty() {
            return VitalityTrend::Stable;
        }

        let oldest = match self.vitality_history.first() {
            Some(entry) => entry.score,
            None => return VitalityTrend::Stable,
        };
        let delta = current - oldest;
        let delta_pct = (delta / oldest) * 100.0;

        match delta_pct {
            d if d > 5.0 => VitalityTrend::Improving,
            d if d < -10.0 => VitalityTrend::RapidDecline,
            d if d < -5.0 => VitalityTrend::Declining,
            _ => VitalityTrend::Stable,
        }
    }

    /// Generate policy adjustment based on current vitality
    pub fn generate_adjustment(&self) -> PolicyAdjustment {
        let current = match self.vitality_history.last() {
            Some(v) => v,
            None => {
                return PolicyAdjustment {
                    fee_rate_factor: 1.0,
                    spore_adjustment: 0,
                    decay_rate_factor: 1.0,
                    velocity_incentive: 1.0,
                    emergency_release: None,
                    reason: "No vitality data available".to_string(),
                    requires_approval: false,
                };
            }
        };

        match current.state {
            MetabolicState::Thriving => self.thriving_adjustment(),
            MetabolicState::Healthy => self.healthy_adjustment(),
            MetabolicState::Stressed => self.stressed_adjustment(),
            MetabolicState::Critical => self.critical_adjustment(),
            MetabolicState::Failing => self.failing_adjustment(),
        }
    }

    fn thriving_adjustment(&self) -> PolicyAdjustment {
        // Increase rewards, maintain fees
        PolicyAdjustment {
            fee_rate_factor: 1.0,
            spore_adjustment: 2,    // Increase SPORE allocation
            decay_rate_factor: 0.9, // Slightly reduce decay
            velocity_incentive: 1.0,
            emergency_release: None,
            reason: "Thriving: Increasing network rewards".to_string(),
            requires_approval: false,
        }
    }

    fn healthy_adjustment(&self) -> PolicyAdjustment {
        // Maintain current parameters
        PolicyAdjustment {
            fee_rate_factor: 1.0,
            spore_adjustment: 0,
            decay_rate_factor: 1.0,
            velocity_incentive: 1.0,
            emergency_release: None,
            reason: "Healthy: Maintaining stable parameters".to_string(),
            requires_approval: false,
        }
    }

    fn stressed_adjustment(&self) -> PolicyAdjustment {
        // Reduce fees, boost circulation
        PolicyAdjustment {
            fee_rate_factor: 0.8, // 20% fee reduction
            spore_adjustment: 0,
            decay_rate_factor: 1.0,
            velocity_incentive: 1.2, // Boost velocity rewards
            emergency_release: None,
            reason: "Stressed: Activating auto-healing".to_string(),
            requires_approval: false,
        }
    }

    fn critical_adjustment(&self) -> PolicyAdjustment {
        // Emergency response
        PolicyAdjustment {
            fee_rate_factor: 0.5,   // 50% fee reduction
            spore_adjustment: 5,    // Boost SPORE
            decay_rate_factor: 0.5, // Reduce decay
            velocity_incentive: 1.5,
            emergency_release: Some(10_000), // Release emergency liquidity
            reason: "Critical: Emergency response activated".to_string(),
            requires_approval: true, // Requires human approval
        }
    }

    fn failing_adjustment(&self) -> PolicyAdjustment {
        // Circuit breaker - minimal activity
        PolicyAdjustment {
            fee_rate_factor: 0.0, // Fee waiver
            spore_adjustment: 0,
            decay_rate_factor: 0.0,          // Suspend decay
            velocity_incentive: 0.0,         // Suspend velocity incentives
            emergency_release: Some(50_000), // Major liquidity release
            reason: "FAILING: Circuit breaker activated - awaiting recovery".to_string(),
            requires_approval: true,
        }
    }

    /// Apply adjustment with bounds checking
    pub fn apply_adjustment(&mut self, adjustment: &PolicyAdjustment) -> Result<(), String> {
        // Check if approval required and not provided
        if adjustment.requires_approval {
            return Err("Adjustment requires Karmic Council approval".to_string());
        }

        // Apply with bounds checking
        let new_fee = self.current_params.fee_rate * adjustment.fee_rate_factor;
        self.current_params.fee_rate =
            new_fee.clamp(self.bounds.fee_rate_min, self.bounds.fee_rate_max);

        let new_spore = (self.current_params.spore_allocation as i64
            + adjustment.spore_adjustment as i64) as u64;
        self.current_params.spore_allocation = new_spore.clamp(
            self.bounds.spore_allocation_min,
            self.bounds.spore_allocation_max,
        );

        let new_decay = self.current_params.decay_rate * adjustment.decay_rate_factor;
        self.current_params.decay_rate =
            new_decay.clamp(self.bounds.decay_rate_min, self.bounds.decay_rate_max);

        self.current_params.velocity_incentive = adjustment.velocity_incentive;

        // Record adjustment for audit
        self.adjustment_history.push(adjustment.clone());

        Ok(())
    }

    /// Get current vitality state
    pub fn current_state(&self) -> Option<MetabolicState> {
        self.vitality_history.last().map(|v| v.state)
    }

    /// Get current vitality score
    pub fn current_vitality(&self) -> Option<f64> {
        self.vitality_history.last().map(|v| v.score)
    }
}

impl Default for MetabolicOracle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_components() -> VitalityComponents {
        VitalityComponents {
            circulation: 0.65,
            relationship: 0.55,
            commons: 0.60,
            resilience: 0.45,
        }
    }

    fn stressed_components() -> VitalityComponents {
        VitalityComponents {
            circulation: 0.30,
            relationship: 0.35,
            commons: 0.25,
            resilience: 0.30,
        }
    }

    #[test]
    fn test_vitality_calculation() {
        let healthy = healthy_components();
        let vitality = healthy.calculate_vitality();

        // 0.65*0.40 + 0.55*0.30 + 0.60*0.20 + 0.45*0.10
        // = 0.26 + 0.165 + 0.12 + 0.045 = 0.59 → 59
        assert!(vitality > 55.0 && vitality < 65.0);
    }

    #[test]
    fn test_state_classification() {
        assert_eq!(MetabolicState::from_score(85.0), MetabolicState::Thriving);
        assert_eq!(MetabolicState::from_score(55.0), MetabolicState::Healthy);
        assert_eq!(MetabolicState::from_score(30.0), MetabolicState::Stressed);
        assert_eq!(MetabolicState::from_score(15.0), MetabolicState::Critical);
        assert_eq!(MetabolicState::from_score(5.0), MetabolicState::Failing);
    }

    #[test]
    fn test_oracle_adjustment_healthy() {
        let mut oracle = MetabolicOracle::new();
        oracle.record_vitality(healthy_components(), 1000);

        let adjustment = oracle.generate_adjustment();
        assert!((adjustment.fee_rate_factor - 1.0).abs() < 0.01);
        assert!(!adjustment.requires_approval);
    }

    #[test]
    fn test_oracle_adjustment_stressed() {
        let mut oracle = MetabolicOracle::new();
        oracle.record_vitality(stressed_components(), 1000);

        let adjustment = oracle.generate_adjustment();
        assert!(adjustment.fee_rate_factor < 1.0); // Fee reduction
        assert!(adjustment.velocity_incentive > 1.0); // Velocity boost
    }

    #[test]
    fn test_bounds_enforcement() {
        let mut oracle = MetabolicOracle::new();
        oracle.current_params.fee_rate = 0.001; // At minimum

        let extreme_adjustment = PolicyAdjustment {
            fee_rate_factor: 0.1, // Would push below minimum
            spore_adjustment: 0,
            decay_rate_factor: 1.0,
            velocity_incentive: 1.0,
            emergency_release: None,
            reason: "Test".to_string(),
            requires_approval: false,
        };

        oracle.apply_adjustment(&extreme_adjustment).unwrap();

        // Should be clamped to minimum
        assert!((oracle.current_params.fee_rate - 0.001).abs() < 0.0001);
    }
}
