// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Metabolic Oracle v2
//!
//! Autopoietic self-regulation with 365-day rolling memory,
//! counter-cyclical TEND expansion, and seasonal awareness.
//!
//! Changes from v1:
//! - Memory: 365 days rolling (was 24 hours)
//! - Removed: spore_allocation (CGC gone)
//! - Added: TendLimitTier for counter-cyclical TEND expansion
//! - Counter-cyclical: When stressed, lower fees + expand TEND limits

use serde::{Deserialize, Serialize};

/// Policy bounds preventing runaway self-modification
/// These are constitutional constraints that cannot be modified by the oracle
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PolicyBounds {
    /// Minimum fee rate (0.01% floor)
    pub fee_rate_min: f64,
    /// Maximum fee rate (0.5% ceiling)
    pub fee_rate_max: f64,
    /// Minimum demurrage rate (1% annual floor)
    pub demurrage_rate_min: f64,
    /// Maximum demurrage rate (5% annual ceiling)
    pub demurrage_rate_max: f64,
    /// Emergency reserve minimum (5%)
    pub emergency_reserve_min: f64,
}

impl Default for PolicyBounds {
    fn default() -> Self {
        Self {
            fee_rate_min: 0.0001,
            fee_rate_max: 0.005,
            demurrage_rate_min: 0.01,
            demurrage_rate_max: 0.05,
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
    /// Commons pool utilization + recognition flow
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
    /// Score 80-100: Network thriving
    Thriving,
    /// Score 60-80: Normal operation
    Healthy,
    /// Score 40-60: Auto-healing activates
    Stressed,
    /// Score 20-40: Emergency response
    Critical,
    /// Score 0-20: Circuit breaker
    Failing,
}

impl MetabolicState {
    /// Determine state from vitality score.
    /// Thresholds aligned with canonical types crate (mycelix_finance_types).
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s >= 80.0 => MetabolicState::Thriving,
            s if s >= 60.0 => MetabolicState::Healthy,
            s if s >= 40.0 => MetabolicState::Stressed,
            s if s >= 20.0 => MetabolicState::Critical,
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

/// TEND limit tier for counter-cyclical expansion (WIR Bank pattern)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TendLimitTier {
    /// Healthy/Thriving: ±40 TEND
    Normal,
    /// Stressed: ±60 TEND
    Elevated,
    /// Critical: ±80 TEND
    High,
    /// Failing: ±120 TEND
    Emergency,
}

impl TendLimitTier {
    /// Get the TEND balance limit for this tier
    pub fn limit(&self) -> i32 {
        match self {
            TendLimitTier::Normal => 40,
            TendLimitTier::Elevated => 60,
            TendLimitTier::High => 80,
            TendLimitTier::Emergency => 120,
        }
    }

    /// Determine tier from metabolic state
    pub fn from_state(state: MetabolicState) -> Self {
        match state {
            MetabolicState::Thriving | MetabolicState::Healthy => TendLimitTier::Normal,
            MetabolicState::Stressed => TendLimitTier::Elevated,
            MetabolicState::Critical => TendLimitTier::High,
            MetabolicState::Failing => TendLimitTier::Emergency,
        }
    }
}

/// Vitality trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VitalityTrend {
    /// Improving over measurement window
    Improving,
    /// Stable (±5%)
    Stable,
    /// Declining over measurement window
    Declining,
    /// Rapid decline (>10%)
    RapidDecline,
}

/// Policy adjustment recommendation from oracle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyAdjustment {
    /// Fee rate adjustment factor (1.0 = no change)
    pub fee_rate_factor: f64,
    /// Demurrage rate adjustment factor
    pub demurrage_rate_factor: f64,
    /// Velocity incentive multiplier
    pub velocity_incentive: f64,
    /// TEND limit tier for counter-cyclical expansion
    pub tend_limit_tier: TendLimitTier,
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
    /// Historical vitality readings (365-day rolling)
    pub vitality_history: Vec<VitalityIndex>,
    /// Adjustment history for audit
    pub adjustment_history: Vec<PolicyAdjustment>,
}

/// Current network economic parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkParameters {
    /// Current base fee rate
    pub fee_rate: f64,
    /// Current demurrage rate (annual)
    pub demurrage_rate: f64,
    /// Velocity incentive multiplier
    pub velocity_incentive: f64,
    /// Emergency reserve ratio
    pub emergency_reserve: f64,
    /// Current TEND limit tier
    pub tend_limit_tier: TendLimitTier,
}

impl Default for NetworkParameters {
    fn default() -> Self {
        Self {
            fee_rate: 0.0003,     // 0.03% (Member tier default)
            demurrage_rate: 0.02, // 2% annual
            velocity_incentive: 1.0,
            emergency_reserve: 0.10,
            tend_limit_tier: TendLimitTier::Normal,
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

        // Keep 365 days of history (assuming daily measurements)
        if self.vitality_history.len() > 365 {
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

        if oldest == 0.0 {
            return VitalityTrend::Stable;
        }

        let delta_pct = ((current - oldest) / oldest) * 100.0;

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
                    demurrage_rate_factor: 1.0,
                    velocity_incentive: 1.0,
                    tend_limit_tier: TendLimitTier::Normal,
                    emergency_release: None,
                    reason: "No vitality data available".to_string(),
                    requires_approval: false,
                }
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
        PolicyAdjustment {
            fee_rate_factor: 1.0,
            demurrage_rate_factor: 0.9,
            velocity_incentive: 1.0,
            tend_limit_tier: TendLimitTier::Normal,
            emergency_release: None,
            reason: "Thriving: Slightly reducing demurrage".to_string(),
            requires_approval: false,
        }
    }

    fn healthy_adjustment(&self) -> PolicyAdjustment {
        PolicyAdjustment {
            fee_rate_factor: 1.0,
            demurrage_rate_factor: 1.0,
            velocity_incentive: 1.0,
            tend_limit_tier: TendLimitTier::Normal,
            emergency_release: None,
            reason: "Healthy: Maintaining stable parameters".to_string(),
            requires_approval: false,
        }
    }

    fn stressed_adjustment(&self) -> PolicyAdjustment {
        // Counter-cyclical: lower fees + expand TEND limits
        PolicyAdjustment {
            fee_rate_factor: 0.8,
            demurrage_rate_factor: 1.0,
            velocity_incentive: 1.2,
            tend_limit_tier: TendLimitTier::Elevated, // ±60 TEND
            emergency_release: None,
            reason: "Stressed: Lower fees, expand TEND limits to ±60".to_string(),
            requires_approval: false,
        }
    }

    fn critical_adjustment(&self) -> PolicyAdjustment {
        PolicyAdjustment {
            fee_rate_factor: 0.5,
            demurrage_rate_factor: 0.5,
            velocity_incentive: 1.5,
            tend_limit_tier: TendLimitTier::High, // ±80 TEND
            emergency_release: Some(10_000),
            reason: "Critical: Emergency response, TEND limits ±80".to_string(),
            requires_approval: true,
        }
    }

    fn failing_adjustment(&self) -> PolicyAdjustment {
        PolicyAdjustment {
            fee_rate_factor: 0.0,
            demurrage_rate_factor: 0.0,
            velocity_incentive: 0.0,
            tend_limit_tier: TendLimitTier::Emergency, // ±120 TEND
            emergency_release: Some(50_000),
            reason: "FAILING: Circuit breaker, TEND limits ±120".to_string(),
            requires_approval: true,
        }
    }

    /// Apply adjustment with bounds checking
    pub fn apply_adjustment(&mut self, adjustment: &PolicyAdjustment) -> Result<(), String> {
        if adjustment.requires_approval {
            return Err("Adjustment requires governance approval".to_string());
        }

        let new_fee = self.current_params.fee_rate * adjustment.fee_rate_factor;
        self.current_params.fee_rate =
            new_fee.clamp(self.bounds.fee_rate_min, self.bounds.fee_rate_max);

        let new_demurrage = self.current_params.demurrage_rate * adjustment.demurrage_rate_factor;
        self.current_params.demurrage_rate = new_demurrage.clamp(
            self.bounds.demurrage_rate_min,
            self.bounds.demurrage_rate_max,
        );

        self.current_params.velocity_incentive = adjustment.velocity_incentive;
        self.current_params.tend_limit_tier = adjustment.tend_limit_tier;

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
        // Target vitality ~70 (Healthy tier: 60-80)
        // 0.75*0.40 + 0.70*0.30 + 0.65*0.20 + 0.55*0.10 = 0.30+0.21+0.13+0.055 = 0.695 → 69.5
        VitalityComponents {
            circulation: 0.75,
            relationship: 0.70,
            commons: 0.65,
            resilience: 0.55,
        }
    }

    fn stressed_components() -> VitalityComponents {
        // Target vitality ~45 (Stressed tier: 40-60)
        // 0.45*0.40 + 0.50*0.30 + 0.40*0.20 + 0.35*0.10 = 0.18+0.15+0.08+0.035 = 0.445 → 44.5
        VitalityComponents {
            circulation: 0.45,
            relationship: 0.50,
            commons: 0.40,
            resilience: 0.35,
        }
    }

    #[test]
    fn test_vitality_calculation() {
        let healthy = healthy_components();
        let vitality = healthy.calculate_vitality();
        // 0.75*0.40 + 0.70*0.30 + 0.65*0.20 + 0.55*0.10 = 69.5
        assert!(
            vitality > 65.0 && vitality < 75.0,
            "Expected ~69.5, got {}",
            vitality
        );
    }

    #[test]
    fn test_state_classification() {
        // Thresholds aligned with mycelix_finance_types::MetabolicState::from_vitality
        assert_eq!(MetabolicState::from_score(85.0), MetabolicState::Thriving);
        assert_eq!(MetabolicState::from_score(80.0), MetabolicState::Thriving);
        assert_eq!(MetabolicState::from_score(79.9), MetabolicState::Healthy);
        assert_eq!(MetabolicState::from_score(60.0), MetabolicState::Healthy);
        assert_eq!(MetabolicState::from_score(55.0), MetabolicState::Stressed);
        assert_eq!(MetabolicState::from_score(40.0), MetabolicState::Stressed);
        assert_eq!(MetabolicState::from_score(30.0), MetabolicState::Critical);
        assert_eq!(MetabolicState::from_score(20.0), MetabolicState::Critical);
        assert_eq!(MetabolicState::from_score(15.0), MetabolicState::Failing);
        assert_eq!(MetabolicState::from_score(5.0), MetabolicState::Failing);
    }

    #[test]
    fn test_tend_limit_tiers() {
        assert_eq!(
            TendLimitTier::from_state(MetabolicState::Healthy).limit(),
            40
        );
        assert_eq!(
            TendLimitTier::from_state(MetabolicState::Stressed).limit(),
            60
        );
        assert_eq!(
            TendLimitTier::from_state(MetabolicState::Critical).limit(),
            80
        );
        assert_eq!(
            TendLimitTier::from_state(MetabolicState::Failing).limit(),
            120
        );
    }

    #[test]
    fn test_oracle_stressed_expands_tend() {
        let mut oracle = MetabolicOracle::new();
        oracle.record_vitality(stressed_components(), 1000);

        let adjustment = oracle.generate_adjustment();
        assert_eq!(adjustment.tend_limit_tier, TendLimitTier::Elevated);
        assert!(adjustment.fee_rate_factor < 1.0);
        assert!(adjustment.velocity_incentive > 1.0);
    }

    #[test]
    fn test_bounds_enforcement() {
        let mut oracle = MetabolicOracle::new();
        oracle.current_params.fee_rate = 0.0001; // At minimum

        let extreme = PolicyAdjustment {
            fee_rate_factor: 0.1,
            demurrage_rate_factor: 1.0,
            velocity_incentive: 1.0,
            tend_limit_tier: TendLimitTier::Normal,
            emergency_release: None,
            reason: "Test".to_string(),
            requires_approval: false,
        };

        oracle.apply_adjustment(&extreme).unwrap();
        assert!((oracle.current_params.fee_rate - 0.0001).abs() < 0.00001);
    }
}
