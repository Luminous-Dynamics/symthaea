// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persistent energy reservoir for consciousness-coupled entities.
//!
//! Energy is a finite, conserved quantity. It depletes through movement,
//! consciousness maintenance, and collisions. It regenerates through
//! harmony resonance, energy wells, and slow ambient recovery.
//!
//! When energy reaches zero, the entity collapses — Φ drops to 0,
//! motor output halts, and recovery requires another agent's help.

/// Persistent energy reservoir for a consciousness-coupled entity.
///
/// Tracks internal energy (U), temperature (T), and entropy (S)
/// for 2nd Law compliance. Available work = Helmholtz free energy F = U - TS.
#[derive(Debug, Clone)]
pub struct EnergyBudget {
    /// Maximum energy capacity.
    pub max_energy: f64,
    /// Current internal energy U (persistent — does NOT reset each tick).
    pub available: f64,
    /// Entity temperature in Kelvin (rises with dissipated heat).
    pub temperature: f64,
    /// Cumulative entropy in J/K (monotonically increases per 2nd Law).
    pub entropy: f64,
    /// Heat capacity in J/K.
    pub heat_capacity: f64,
    /// Total energy consumed this tick (reset each tick for telemetry).
    pub consumed_this_tick: f64,
    /// Total energy regenerated this tick (reset each tick for telemetry).
    pub regenerated_this_tick: f64,
    /// Cumulative energy spent across all ticks.
    pub lifetime_consumed: f64,
    /// Whether this entity has collapsed (energy depleted to zero).
    pub collapsed: bool,
}

impl EnergyBudget {
    /// Create a new energy budget with starting energy.
    pub fn new(max_energy: f64) -> Self {
        Self {
            max_energy,
            available: max_energy,
            temperature: 310.0,
            entropy: 0.0,
            heat_capacity: 100.0,
            consumed_this_tick: 0.0,
            regenerated_this_tick: 0.0,
            lifetime_consumed: 0.0,
            collapsed: false,
        }
    }

    /// Reset per-tick counters (call at start of each tick).
    pub fn tick_reset(&mut self) {
        self.consumed_this_tick = 0.0;
        self.regenerated_this_tick = 0.0;
    }

    /// Try to consume energy. Returns the amount actually consumed
    /// (may be less than requested if budget is exhausted).
    #[inline]
    pub fn consume(&mut self, amount: f64) -> f64 {
        if self.collapsed {
            return 0.0;
        }
        let actual = amount.min(self.available);
        self.available -= actual;
        self.consumed_this_tick += actual;
        self.lifetime_consumed += actual;

        // 2nd Law: irreversible consumption increases entropy
        if self.temperature > 0.0 && actual > 0.0 {
            self.entropy += actual / self.temperature;
        }

        // Check for collapse
        if self.available <= 0.0 {
            self.available = 0.0;
            self.collapsed = true;
        }
        actual
    }

    /// Absorb heat from dissipation (damping, friction, collision).
    /// Increases temperature and entropy per 2nd Law.
    pub fn dissipate_heat(&mut self, energy: f64) {
        if energy <= 0.0 || self.heat_capacity <= 0.0 {
            return;
        }
        self.temperature += energy / self.heat_capacity;
        self.entropy += energy / self.temperature;
    }

    /// Helmholtz free energy: maximum extractable work at constant T.
    /// F = U - TS. Always ≥ 0.
    pub fn available_work(&self) -> f64 {
        (self.available - self.temperature * self.entropy).max(0.0)
    }

    /// Add energy (from regeneration sources). Capped at max_energy.
    #[inline]
    pub fn regenerate(&mut self, amount: f64) {
        self.available = (self.available + amount).min(self.max_energy);
        self.regenerated_this_tick += amount;

        // Recover from collapse if energy is restored
        if self.collapsed && self.available > 0.0 {
            self.collapsed = false;
        }
    }

    /// Whether any energy is available.
    #[inline]
    pub fn has_energy(&self) -> bool {
        self.available > 1e-10 && !self.collapsed
    }

    /// Whether this entity has collapsed (zero energy).
    #[inline]
    pub fn is_collapsed(&self) -> bool {
        self.collapsed
    }

    /// Fraction of energy remaining [0.0, 1.0].
    #[inline]
    pub fn fraction_remaining(&self) -> f64 {
        if self.max_energy < 1e-10 {
            return 0.0;
        }
        self.available / self.max_energy
    }

    /// Net energy flow this tick (positive = regenerating, negative = depleting).
    pub fn net_flow_this_tick(&self) -> f64 {
        self.regenerated_this_tick - self.consumed_this_tick
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_budget_has_full_energy() {
        let budget = EnergyBudget::new(100.0);
        assert!((budget.available - 100.0).abs() < 1e-10);
        assert!(!budget.collapsed);
    }

    #[test]
    fn consume_reduces_available() {
        let mut budget = EnergyBudget::new(100.0);
        let consumed = budget.consume(30.0);
        assert!((consumed - 30.0).abs() < 1e-10);
        assert!((budget.available - 70.0).abs() < 1e-10);
    }

    #[test]
    fn consume_capped_by_available() {
        let mut budget = EnergyBudget::new(100.0);
        budget.consume(80.0);
        let consumed = budget.consume(50.0); // only 20 left
        assert!((consumed - 20.0).abs() < 1e-10);
        assert!(budget.is_collapsed());
    }

    #[test]
    fn collapse_on_zero_energy() {
        let mut budget = EnergyBudget::new(100.0);
        budget.consume(100.0);
        assert!(budget.is_collapsed());
        assert!(!budget.has_energy());

        // Collapsed entity cannot consume
        let consumed = budget.consume(10.0);
        assert!((consumed - 0.0).abs() < 1e-10);
    }

    #[test]
    fn regenerate_adds_energy() {
        let mut budget = EnergyBudget::new(100.0);
        budget.consume(50.0);
        budget.regenerate(20.0);
        assert!((budget.available - 70.0).abs() < 1e-10);
    }

    #[test]
    fn regenerate_capped_at_max() {
        let mut budget = EnergyBudget::new(100.0);
        budget.regenerate(50.0); // already at max
        assert!((budget.available - 100.0).abs() < 1e-10);
    }

    #[test]
    fn regenerate_recovers_from_collapse() {
        let mut budget = EnergyBudget::new(100.0);
        budget.consume(100.0);
        assert!(budget.is_collapsed());

        budget.regenerate(10.0);
        assert!(!budget.is_collapsed());
        assert!(budget.has_energy());
    }

    #[test]
    fn lifetime_accumulates() {
        let mut budget = EnergyBudget::new(100.0);
        budget.consume(30.0);
        budget.tick_reset();
        budget.consume(20.0);
        assert!((budget.lifetime_consumed - 50.0).abs() < 1e-10);
    }

    #[test]
    fn net_flow_tracks_correctly() {
        let mut budget = EnergyBudget::new(100.0);
        budget.consume(10.0);
        budget.regenerate(15.0);
        assert!((budget.net_flow_this_tick() - 5.0).abs() < 1e-10); // +5 net
    }

    #[test]
    fn tick_reset_clears_per_tick() {
        let mut budget = EnergyBudget::new(100.0);
        budget.consume(10.0);
        budget.regenerate(5.0);
        budget.tick_reset();
        assert!((budget.consumed_this_tick - 0.0).abs() < 1e-10);
        assert!((budget.regenerated_this_tick - 0.0).abs() < 1e-10);
        // But available is unchanged
        assert!((budget.available - 95.0).abs() < 1e-10);
    }

    #[test]
    fn fraction_remaining() {
        let mut budget = EnergyBudget::new(100.0);
        budget.consume(25.0);
        assert!((budget.fraction_remaining() - 0.75).abs() < 1e-10);
    }
}
