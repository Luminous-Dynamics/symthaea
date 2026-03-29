//! Consciousness-gated energy budgets.
//!
//! Each entity has an energy budget per physics tick. Higher consciousness
//! unlocks more energy for actions (thermodynamic/FEP alignment: consciousness
//! is the capacity to do useful thermodynamic work).

use crate::safety::SafetyTier;

/// Energy budget for a consciousness-coupled entity.
#[derive(Debug, Clone)]
pub struct EnergyBudget {
    /// Maximum energy available per tick at full consciousness (Green tier).
    pub max_energy: f64,
    /// Current available energy this tick.
    pub available: f64,
    /// Total energy consumed this tick.
    pub consumed: f64,
    /// Cumulative energy spent across all ticks.
    pub lifetime_consumed: f64,
}

impl EnergyBudget {
    /// Create a new energy budget with the given maximum.
    pub fn new(max_energy: f64) -> Self {
        Self {
            max_energy,
            available: max_energy,
            consumed: 0.0,
            lifetime_consumed: 0.0,
        }
    }

    /// Refresh the energy budget for a new tick based on consciousness level.
    ///
    /// The available energy scales with Φ:
    /// - Φ > 0.6 (Green): 100% of max
    /// - Φ 0.3–0.6 (Yellow): 60% of max
    /// - Φ 0.1–0.3 (Orange): 30% of max
    /// - Φ ≤ 0.1 (Red): 0% (shutdown)
    pub fn refresh(&mut self, phi: f64) {
        let tier = SafetyTier::from_phi(phi);
        self.available = self.max_energy * tier.motor_gain();
        self.consumed = 0.0;
    }

    /// Try to consume energy. Returns the amount actually consumed
    /// (may be less than requested if budget is exhausted).
    #[inline]
    pub fn consume(&mut self, amount: f64) -> f64 {
        let actual = amount.min(self.available);
        self.available -= actual;
        self.consumed += actual;
        self.lifetime_consumed += actual;
        actual
    }

    /// Whether any energy is available.
    #[inline]
    pub fn has_energy(&self) -> bool {
        self.available > 1e-10
    }

    /// Fraction of budget remaining [0.0, 1.0].
    #[inline]
    pub fn fraction_remaining(&self) -> f64 {
        if self.max_energy < 1e-10 {
            return 0.0;
        }
        self.available / self.max_energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_consciousness_full_energy() {
        let mut budget = EnergyBudget::new(100.0);
        budget.refresh(1.0); // Green tier
        assert!((budget.available - 100.0).abs() < 1e-10);
    }

    #[test]
    fn low_consciousness_low_energy() {
        let mut budget = EnergyBudget::new(100.0);
        budget.refresh(0.4); // Yellow tier
        assert!((budget.available - 60.0).abs() < 1e-10);
    }

    #[test]
    fn red_tier_no_energy() {
        let mut budget = EnergyBudget::new(100.0);
        budget.refresh(0.05); // Red tier
        assert!(budget.available < 1e-10);
        assert!(!budget.has_energy());
    }

    #[test]
    fn consume_tracks_usage() {
        let mut budget = EnergyBudget::new(100.0);
        budget.refresh(1.0);
        let used = budget.consume(30.0);
        assert!((used - 30.0).abs() < 1e-10);
        assert!((budget.available - 70.0).abs() < 1e-10);
        assert!((budget.consumed - 30.0).abs() < 1e-10);
    }

    #[test]
    fn consume_capped_by_available() {
        let mut budget = EnergyBudget::new(100.0);
        budget.refresh(0.4); // 60 available
        let used = budget.consume(80.0); // request 80, only 60 available
        assert!((used - 60.0).abs() < 1e-10);
        assert!(!budget.has_energy());
    }

    #[test]
    fn lifetime_accumulates() {
        let mut budget = EnergyBudget::new(100.0);

        budget.refresh(1.0);
        budget.consume(30.0);

        budget.refresh(1.0);
        budget.consume(20.0);

        assert!((budget.lifetime_consumed - 50.0).abs() < 1e-10);
    }

    #[test]
    fn fraction_remaining() {
        let mut budget = EnergyBudget::new(100.0);
        budget.refresh(1.0);
        budget.consume(25.0);
        assert!((budget.fraction_remaining() - 0.75).abs() < 1e-10);
    }
}
