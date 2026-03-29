// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Thermodynamic closure: unifying consciousness energy and physics energy.
//!
//! This module implements the novel Joules-per-Phi energy accounting system.
//! Every action costs physical energy from the consciousness budget. Collision
//! impulses dissipate energy as heat. The total energy is tracked and conserved.
//!
//! # Key Quantities
//!
//! - **Landauer bound**: minimum energy per bit of information processing
//!   (k_B * T * ln(2) ≈ 2.87 × 10⁻²¹ J at body temperature 310K)
//! - **Joules-per-Phi**: novel metric — how many Joules does it cost to
//!   maintain a given Φ level? No one has published this.
//! - **Conservation error**: measures violation of energy conservation law.
//!   Should remain < 1% for a valid simulation.

/// Landauer bound at body temperature (310K): k_B * T * ln(2).
/// Minimum energy to erase one bit of information.
/// Ref: Landauer (1961), "Irreversibility and Heat Generation"
pub const LANDAUER_BOUND_310K: f64 = 2.87e-21; // Joules per bit

/// Default energy per cognitive operation (empirical estimate).
/// Cortical synapses operate at ~16.6× Landauer limit.
/// Ref: Laughlin & Sejnowski (2003)
pub const ENERGY_PER_COGNITIVE_OP: f64 = LANDAUER_BOUND_310K * 16.6;

/// Tunable thermodynamic constants for the enforcement layer.
///
/// These values determine how quickly energy depletes and regenerates.
/// Calibrated so that solo play survives ~4 minutes, cooperation sustains indefinitely.
#[derive(Debug, Clone)]
pub struct ThermodynamicConstants {
    /// Starting Joules for a new entity.
    pub initial_energy: f64,
    /// Maximum energy an entity can hold.
    pub max_energy: f64,
    /// Joules per physics-unit of displacement (walking).
    /// At WALK_SPEED=100 u/s, 64Hz: ~0.0078 J/tick for movement.
    pub movement_cost_per_unit: f64,
    /// Sprint cost multiplier (applied on top of movement_cost_per_unit).
    pub sprint_cost_multiplier: f64,
    /// Joules per tick to maintain consciousness (Φ > 0).
    /// Higher Φ costs more: actual = base * (1.0 + phi * 0.5).
    pub consciousness_maintenance_per_tick: f64,
    /// Fraction of collision impulse magnitude drained as energy.
    pub collision_energy_drain: f64,
    /// Joules per tick gained when harmony resonance > 0.5 with a nearby agent.
    pub harmony_resonance_regen_rate: f64,
    /// Joules per tick gained when standing in an energy well.
    pub energy_well_regen_rate: f64,
    /// Slow background regeneration (too slow alone).
    pub ambient_regen_rate: f64,
    /// Harmony resonance threshold to rescue a collapsed entity.
    pub collapse_recovery_harmony_threshold: f64,
    /// Range within which harmony resonance transfers energy.
    pub harmony_range: f64,
}

impl Default for ThermodynamicConstants {
    fn default() -> Self {
        Self {
            initial_energy: 1000.0,
            max_energy: 1000.0,
            movement_cost_per_unit: 0.005,
            sprint_cost_multiplier: 2.5,
            consciousness_maintenance_per_tick: 0.08,
            collision_energy_drain: 0.05,
            harmony_resonance_regen_rate: 0.15,
            energy_well_regen_rate: 0.25,
            ambient_regen_rate: 0.02,
            collapse_recovery_harmony_threshold: 0.5,
            harmony_range: 50.0,
        }
    }
}

/// Thermodynamic ledger: tracks energy flow through the consciousness-physics system.
#[derive(Debug, Clone)]
pub struct ThermodynamicLedger {
    /// Total energy input this tick (forces applied, work done).
    pub energy_in: f64,
    /// Total energy output this tick (dissipated as heat via damping/friction).
    pub energy_out: f64,
    /// Phi-weighted energy integral: Σ(energy_i × phi_i) across all entities.
    /// Used to compute Joules-per-Phi.
    pub phi_energy_integral: f64,
    /// Total Phi change this tick: Σ|Δphi_i|.
    pub phi_change_total: f64,
    /// Cumulative energy consumed across all ticks.
    pub lifetime_energy: f64,
    /// Cumulative conservation error across all ticks.
    pub lifetime_error: f64,
    /// Number of ticks processed.
    pub tick_count: u64,
}

impl ThermodynamicLedger {
    pub fn new() -> Self {
        Self {
            energy_in: 0.0,
            energy_out: 0.0,
            phi_energy_integral: 0.0,
            phi_change_total: 0.0,
            lifetime_energy: 0.0,
            lifetime_error: 0.0,
            tick_count: 0,
        }
    }

    /// Record an energy-consuming action.
    ///
    /// `energy_cost` is the Joules consumed.
    /// `phi` is the entity's current consciousness level.
    pub fn record_action(&mut self, energy_cost: f64, phi: f64) {
        self.energy_in += energy_cost;
        self.phi_energy_integral += energy_cost * phi;
    }

    /// Record energy dissipated (friction, damping, collision heat).
    pub fn record_dissipation(&mut self, energy: f64) {
        self.energy_out += energy;
    }

    /// Record a change in Phi for an entity.
    pub fn record_phi_change(&mut self, delta_phi: f64) {
        self.phi_change_total += delta_phi.abs();
    }

    /// Finalize this tick: compute conservation error and accumulate.
    pub fn tick_balance(&mut self) -> TickBalance {
        let balance = self.energy_in - self.energy_out;
        let error = balance.abs();
        self.lifetime_energy += self.energy_in;
        self.lifetime_error += error;
        self.tick_count += 1;

        let result = TickBalance {
            energy_in: self.energy_in,
            energy_out: self.energy_out,
            balance,
            conservation_error: if self.energy_in > 1e-15 {
                error / self.energy_in
            } else {
                0.0
            },
            joules_per_phi: self.joules_per_phi(),
        };

        // Reset per-tick accumulators
        self.energy_in = 0.0;
        self.energy_out = 0.0;
        self.phi_energy_integral = 0.0;
        self.phi_change_total = 0.0;

        result
    }

    /// Novel metric: Joules per unit Phi.
    ///
    /// How much energy does it cost to maintain consciousness?
    /// Returns None if no Phi change this tick.
    pub fn joules_per_phi(&self) -> Option<f64> {
        if self.phi_change_total < 1e-15 {
            return None;
        }
        Some(self.phi_energy_integral / self.phi_change_total)
    }

    /// Minimum thermodynamic cost of processing `bits` of information.
    /// Based on the Landauer bound at body temperature.
    pub fn landauer_floor(bits: f64) -> f64 {
        LANDAUER_BOUND_310K * bits
    }

    /// Whether an energy cost exceeds the Landauer floor for given bits processed.
    /// If the energy is below the floor, the computation is thermodynamically impossible.
    pub fn above_landauer_floor(energy: f64, bits: f64) -> bool {
        energy >= Self::landauer_floor(bits)
    }

    /// Lifetime conservation error rate.
    pub fn lifetime_error_rate(&self) -> f64 {
        if self.lifetime_energy < 1e-15 {
            return 0.0;
        }
        self.lifetime_error / self.lifetime_energy
    }
}

impl Default for ThermodynamicLedger {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a single tick's energy balance.
#[derive(Debug, Clone)]
pub struct TickBalance {
    /// Total energy consumed this tick.
    pub energy_in: f64,
    /// Total energy dissipated this tick.
    pub energy_out: f64,
    /// Net energy balance (should be near zero for conservation).
    pub balance: f64,
    /// Fractional conservation error [0, 1].
    pub conservation_error: f64,
    /// Joules per unit Phi this tick (None if no Phi change).
    pub joules_per_phi: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_ledger_zero_error() {
        let mut ledger = ThermodynamicLedger::new();
        let balance = ledger.tick_balance();
        assert_eq!(balance.conservation_error, 0.0);
    }

    #[test]
    fn balanced_energy_zero_error() {
        let mut ledger = ThermodynamicLedger::new();
        ledger.record_action(100.0, 0.5);
        ledger.record_dissipation(100.0);
        let balance = ledger.tick_balance();
        assert!(
            balance.conservation_error < 1e-10,
            "error = {}",
            balance.conservation_error
        );
    }

    #[test]
    fn unbalanced_energy_nonzero_error() {
        let mut ledger = ThermodynamicLedger::new();
        ledger.record_action(100.0, 0.5);
        ledger.record_dissipation(80.0); // 20% lost
        let balance = ledger.tick_balance();
        assert!(
            (balance.conservation_error - 0.2).abs() < 1e-10,
            "error = {}",
            balance.conservation_error
        );
    }

    #[test]
    fn joules_per_phi() {
        let mut ledger = ThermodynamicLedger::new();
        // Entity with phi=0.8 consumes 100J
        ledger.record_action(100.0, 0.8);
        ledger.record_phi_change(0.2); // Phi changed by 0.2
        let j_per_phi = ledger.joules_per_phi().unwrap();
        // phi_energy_integral = 100 * 0.8 = 80
        // joules_per_phi = 80 / 0.2 = 400
        assert!(
            (j_per_phi - 400.0).abs() < 1e-10,
            "j/phi = {}",
            j_per_phi
        );
    }

    #[test]
    fn no_phi_change_returns_none() {
        let mut ledger = ThermodynamicLedger::new();
        ledger.record_action(100.0, 0.5);
        assert!(ledger.joules_per_phi().is_none());
    }

    #[test]
    fn landauer_bound() {
        let floor = ThermodynamicLedger::landauer_floor(1.0);
        assert!((floor - LANDAUER_BOUND_310K).abs() < 1e-30);

        // 1 billion bits
        let floor_1g = ThermodynamicLedger::landauer_floor(1e9);
        assert!(floor_1g > 1e-12); // ~2.87e-12 J for 1 Gbit
    }

    #[test]
    fn above_landauer_floor_check() {
        assert!(ThermodynamicLedger::above_landauer_floor(1e-20, 1.0)); // Way above
        assert!(!ThermodynamicLedger::above_landauer_floor(1e-22, 1.0)); // Below
    }

    #[test]
    fn lifetime_accumulates() {
        let mut ledger = ThermodynamicLedger::new();

        ledger.record_action(100.0, 0.5);
        ledger.record_dissipation(100.0);
        ledger.tick_balance();

        ledger.record_action(200.0, 0.7);
        ledger.record_dissipation(200.0);
        ledger.tick_balance();

        assert!((ledger.lifetime_energy - 300.0).abs() < 1e-10);
        assert_eq!(ledger.tick_count, 2);
    }

    #[test]
    fn tick_resets_accumulators() {
        let mut ledger = ThermodynamicLedger::new();
        ledger.record_action(100.0, 0.5);
        ledger.tick_balance();

        // After tick, per-tick values should be reset
        assert!(ledger.energy_in < 1e-15);
        assert!(ledger.energy_out < 1e-15);
    }

    #[test]
    fn lifetime_error_rate() {
        let mut ledger = ThermodynamicLedger::new();
        ledger.record_action(100.0, 0.5);
        ledger.record_dissipation(90.0); // 10% error
        ledger.tick_balance();

        let rate = ledger.lifetime_error_rate();
        assert!(
            (rate - 0.1).abs() < 1e-10,
            "error rate = {}",
            rate
        );
    }

    #[test]
    fn energy_per_cognitive_op() {
        // Verify the constant is physically reasonable
        // Brain uses ~20W for ~10^11 synapses at ~10Hz = 10^12 ops/sec
        // 20W / 10^12 = 2×10^-11 J/op
        // Our constant: ~4.8×10^-20 J/op (much lower — Landauer-based, not metabolic)
        assert!(ENERGY_PER_COGNITIVE_OP > 0.0);
        assert!(ENERGY_PER_COGNITIVE_OP < 1e-15); // Not absurdly large
    }
}
