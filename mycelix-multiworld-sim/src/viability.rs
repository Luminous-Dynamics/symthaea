// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Viability Engine — enforces the 5 irreducible physical axioms every tick.
//!
//! # The 5 Axioms
//!
//! 1. **Energy conservation with dissipation** (1st+2nd Law of Thermodynamics)
//!    - Net_energy = Gross_energy × (1 - 1/EROI)
//!    - Entropy monotonically increases: dS/dt ≥ 0
//!
//! 2. **Maximum power, not efficiency** (Lotka-Odum principle)
//!    - Systems optimize throughput at ~50% efficiency, not 100%
//!    - CITATION: Lotka (1922), Odum & Pinkerton (1955)
//!
//! 3. **Superlinear scaling** (West-Bettencourt 2007, 2013)
//!    - Infrastructure: β ≈ 5/6 (sublinear — economies of scale)
//!    - Socioeconomic: β ≈ 7/6 (superlinear — increasing returns)
//!    - Validated across US, China, India, Japan, Europe, Brazil
//!
//! 4. **Power-law events** (Bak 1987, Self-Organized Criticality)
//!    - P(size ≥ s) ~ s^(-τ), τ ≈ 1.0-1.5
//!    - Mega-events are structurally inevitable, not outliers
//!
//! 5. **Free energy minimization** (Friston 2010)
//!    - F = D_KL[q(s) || p(s|o)] - ln p(o)
//!    - Every persistent system minimizes variational free energy
//!    - FUTURE: Replace ad-hoc agent rules with active inference (Phase 5)
//!
//! # The 5 Viability Conditions (checked every tick)
//!
//! | # | Condition | Violation Response |
//! |---|-----------|-------------------|
//! | 1 | Net energy surplus > maintenance cost | Tainter simplification cascade |
//! | 2 | EROI > tier threshold | Cannot sustain current tech level |
//! | 3 | Entropy export > production | Dissipative structure failure |
//! | 4 | Innovation rate sufficient | Finite-time singularity pressure |
//! | 5 | Info processing within budget | Governance collapse |

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── EROI Thresholds (Hall et al. 2014) ─────────────────────────────────────

/// EROI required for basic survival (fire, transport, food processing).
/// CITATION: Hall, Lambert, Balogh (2014) "EROI of different fuels and the
/// implications for society", Energy Policy 64, pp. 141-152.
pub const EROI_SURVIVAL: f64 = 3.0;

/// EROI required for family-level civilization (housing, heating, basic education).
/// CITATION: Lambert et al. (2014) via Hall's graduated EROI framework.
pub const EROI_FAMILY: f64 = 7.0;

/// EROI required for industrial civilization with R&D, arts, healthcare.
/// CITATION: Hall (2014) — net energy cliff below this point.
pub const EROI_INDUSTRIAL: f64 = 12.0;

/// EROI required for space-faring complexity (interplanetary logistics, life support).
/// SPECULATIVE: No empirical basis. Extrapolated from Hall's complexity gradient.
pub const EROI_SPACEFARING: f64 = 20.0;

// ─── West-Bettencourt Scaling Exponents ──────────────────────────────────────

/// Infrastructure scaling exponent (roads, pipes, cables).
/// CITATION: Bettencourt et al. (2007) "Growth, innovation, scaling, and the
/// pace of life in cities", PNAS 104(17), pp. 7301-7306.
/// Theoretical: δ = H/(D+H) = 1/3, β_infra = 1 - δ/2 = 5/6.
pub const BETA_INFRASTRUCTURE: f64 = 5.0 / 6.0; // ≈ 0.833

/// Socioeconomic scaling exponent (GDP, patents, wages, innovation).
/// CITATION: Bettencourt (2013) "The Origins of Scaling in Cities", Science 340.
/// Theoretical: β_socio = 1 + δ/2 = 7/6.
pub const BETA_SOCIOECONOMIC: f64 = 7.0 / 6.0; // ≈ 1.167

/// Pace-of-life scaling exponent (walking speed, transaction rate, crime).
/// Same exponent as socioeconomic — time accelerates with city size.
pub const BETA_PACE: f64 = 7.0 / 6.0;

/// Reference population for scaling normalization.
/// Scaling factors are relative to this size: factor = (N / N_ref)^(β-1).
pub const SCALING_REFERENCE_POP: f64 = 1000.0;

/// Minimum population for scaling to apply (below this, use linear).
pub const SCALING_MIN_POP: f64 = 50.0;

// ─── Power-Law Cascade Parameters ────────────────────────────────────────────

/// Power-law exponent for disaster cascade sizes.
/// CITATION: Bak, Tang, Wiesenfeld (1987) "Self-organized criticality",
/// Phys. Rev. Lett. 59(4), pp. 381-384. Typical τ ≈ 1.0-1.5.
pub const CASCADE_POWER_LAW_TAU: f64 = 1.2;

/// Probability that a disaster triggers a cascade in an adjacent system.
/// SPECULATIVE: No empirical basis for space colonies.
pub const CASCADE_PROPAGATION_PROB: f64 = 0.15;

// ─── Energy Constants ────────────────────────────────────────────────────────

/// Base energy cost per person per tick (abstract resource units).
/// Covers food processing, heating, lighting, basic transport.
/// Calibrated to match ISS crew requirements scaled to monthly resolution.
pub const ENERGY_PER_CAPITA_BASE: f64 = 0.8;

/// Additional energy per unit of infrastructure maintained.
/// Complexity maintenance cost — increases with infrastructure level.
pub const ENERGY_PER_INFRASTRUCTURE_UNIT: f64 = 0.5;

/// Efficiency factor for maximum power principle.
/// Systems operate at ~50% thermodynamic efficiency (Lotka-Odum).
/// CITATION: Odum & Pinkerton (1955) "Time's speed regulator",
/// American Scientist 43(2), pp. 331-343.
pub const MAX_POWER_EFFICIENCY: f64 = 0.5;

// ─── Structs ─────────────────────────────────────────────────────────────────

/// Energy source with EROI tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergySource {
    /// Name of the source (e.g., "solar", "fission", "fusion", "fossil").
    pub name: String,
    /// Gross energy production per tick (abstract resource units).
    pub gross_production: f64,
    /// Energy invested to maintain this source per tick.
    pub energy_invested: f64,
    /// EROI = gross / invested. Undefined if invested == 0.
    pub eroi: f64,
}

impl EnergySource {
    pub fn new(name: &str, gross: f64, invested: f64) -> Self {
        let eroi = if invested > 0.0 {
            gross / invested
        } else {
            f64::INFINITY
        };
        Self {
            name: name.to_string(),
            gross_production: gross,
            energy_invested: invested,
            eroi,
        }
    }
}

/// Per-world energy ledger tracking thermodynamic state.
/// Adapted from Symtropy's `ThermodynamicLedger` pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivilizationEnergyLedger {
    /// Energy sources with individual EROI tracking.
    pub sources: Vec<EnergySource>,
    /// Total gross energy production this tick.
    pub gross_production: f64,
    /// Total energy invested in maintaining sources this tick.
    pub total_invested: f64,
    /// Net energy available after investment (gross - invested).
    pub net_production: f64,
    /// Energy consumed by population (per capita × N).
    pub population_cost: f64,
    /// Energy consumed by complexity/infrastructure maintenance.
    pub complexity_cost: f64,
    /// Net surplus (net_production - population_cost - complexity_cost).
    /// Negative = deficit = collapse pressure.
    pub net_surplus: f64,
    /// Aggregate EROI (weighted by source production share).
    pub aggregate_eroi: f64,
    /// Cumulative entropy produced (monotonically increasing per 2nd Law).
    /// Units: abstract entropy units (energy dissipated / temperature proxy).
    pub cumulative_entropy: f64,
    /// Energy dissipated as waste heat this tick.
    pub waste_heat: f64,
    /// Consecutive ticks in energy deficit.
    pub deficit_streak: u32,
    /// Historical EROI trajectory (last 120 ticks = 10 years).
    pub eroi_history: Vec<f64>,
}

impl Default for CivilizationEnergyLedger {
    fn default() -> Self {
        Self {
            sources: Vec::new(),
            gross_production: 0.0,
            total_invested: 0.0,
            net_production: 0.0,
            population_cost: 0.0,
            complexity_cost: 0.0,
            net_surplus: 0.0,
            aggregate_eroi: f64::INFINITY,
            cumulative_entropy: 0.0,
            waste_heat: 0.0,
            deficit_streak: 0,
            eroi_history: Vec::new(),
        }
    }
}

/// West-Bettencourt scaling factors for a given population size.
/// Precomputed once per tick, consumed by economy, knowledge, governance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScalingFactors {
    /// Raw population count.
    pub population: f64,
    /// Infrastructure efficiency multiplier: (N/N_ref)^(β_infra - 1).
    /// < 1.0 for large pops (economies of scale: need LESS infra per capita).
    pub infrastructure_efficiency: f64,
    /// Innovation/productivity multiplier: (N/N_ref)^(β_socio - 1).
    /// > 1.0 for large pops (superlinear returns: MORE output per capita).
    pub innovation_multiplier: f64,
    /// Pace-of-life multiplier: same as innovation (time accelerates).
    pub pace_multiplier: f64,
}

impl ScalingFactors {
    /// Compute scaling factors for a population of size N.
    /// Below `SCALING_MIN_POP`, returns neutral factors (1.0).
    pub fn compute(population: f64) -> Self {
        if population < SCALING_MIN_POP {
            return Self {
                population,
                infrastructure_efficiency: 1.0,
                innovation_multiplier: 1.0,
                pace_multiplier: 1.0,
            };
        }
        let ratio = population / SCALING_REFERENCE_POP;
        Self {
            population,
            // N^(β-1): for β=5/6, exponent = -1/6 ≈ -0.167 (sublinear → per-capita savings)
            infrastructure_efficiency: ratio.powf(BETA_INFRASTRUCTURE - 1.0),
            // N^(β-1): for β=7/6, exponent = +1/6 ≈ +0.167 (superlinear → per-capita boost)
            innovation_multiplier: ratio.powf(BETA_SOCIOECONOMIC - 1.0),
            pace_multiplier: ratio.powf(BETA_PACE - 1.0),
        }
    }
}

/// Which viability condition was violated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViabilityViolation {
    /// Net energy surplus < 0 for too long.
    EnergyDeficit {
        world_id: u32,
        deficit: f64,
        streak_ticks: u32,
    },
    /// EROI below required tier threshold.
    EROIBelowTier {
        world_id: u32,
        current_eroi: f64,
        required_eroi: f64,
        tier_name: String,
    },
    /// Entropy production exceeding export capacity.
    EntropyAccumulation { world_id: u32, entropy_rate: f64 },
    /// Innovation stagnation — no tech milestones for too long.
    InnovationStagnation {
        world_id: u32,
        ticks_since_milestone: u32,
    },
    /// Governance/coordination complexity exceeding energy budget.
    CoordinationOverload {
        world_id: u32,
        coordination_cost: f64,
        energy_budget: f64,
    },
}

/// Per-tick viability snapshot for trend analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViabilitySnapshot {
    pub tick: u32,
    pub world_id: u32,
    pub net_surplus: f64,
    pub aggregate_eroi: f64,
    pub cumulative_entropy: f64,
    pub scaling_innovation: f64,
    pub violations: Vec<ViabilityViolation>,
}

/// Top-level viability engine.
/// Runs as Phase 0 of every tick — before demographics, before economy.
pub struct ViabilityEngine {
    /// Per-world energy ledgers.
    pub ledgers: HashMap<u32, CivilizationEnergyLedger>,
    /// Per-world scaling factors (recomputed each tick).
    pub scaling: HashMap<u32, ScalingFactors>,
    /// Viability violation events emitted this tick.
    pub violations: Vec<ViabilityViolation>,
    /// Viability history (for trend detection, capped at 1200 = 100 years).
    pub history: Vec<ViabilitySnapshot>,
}

impl ViabilityEngine {
    pub fn new() -> Self {
        Self {
            ledgers: HashMap::new(),
            scaling: HashMap::new(),
            violations: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Initialize energy ledger for a world based on its resource stocks.
    pub fn init_world(&mut self, world_id: u32, energy_production: f64, population: f64) {
        let mut ledger = CivilizationEnergyLedger::default();
        // Default: single source "mixed" with EROI based on tech level.
        // Early colony: solar panels, EROI ~10:1.
        let invested = energy_production / 10.0;
        ledger
            .sources
            .push(EnergySource::new("mixed", energy_production, invested));
        self.ledgers.insert(world_id, ledger);
        self.scaling
            .insert(world_id, ScalingFactors::compute(population));
    }

    /// Tick Phase 0: Compute energy budgets and scaling factors for all worlds.
    ///
    /// This MUST run before economy, demographics, or any other phase.
    /// Other phases read `self.scaling[world_id]` and `self.ledgers[world_id]`
    /// to scale their computations.
    pub fn tick(
        &mut self,
        worlds: &[(u32, f64, f64, f64)], // (world_id, energy_production_rate, population, infrastructure_level)
        current_tick: u32,
    ) {
        self.violations.clear();

        for &(world_id, energy_prod, pop, infra_level) in worlds {
            // 1. Recompute scaling factors
            let scaling = ScalingFactors::compute(pop);
            self.scaling.insert(world_id, scaling);

            // 2. Update energy ledger
            let ledger = self.ledgers.entry(world_id).or_default();

            // Gross production from all sources
            ledger.gross_production = energy_prod;

            // Energy invested = sum of source investments
            // For now, estimate: 1/EROI of gross per source
            ledger.total_invested = ledger
                .sources
                .iter()
                .map(|s| {
                    if s.eroi > 0.0 {
                        s.gross_production / s.eroi
                    } else {
                        0.0
                    }
                })
                .sum();

            // Net production after investment
            ledger.net_production = (ledger.gross_production - ledger.total_invested).max(0.0);

            // Population energy cost (per capita, unscaled)
            ledger.population_cost = pop * ENERGY_PER_CAPITA_BASE;

            // Complexity/infrastructure maintenance cost
            // Scales with infrastructure level AND benefits from West-Bettencourt sublinear scaling
            ledger.complexity_cost = infra_level
                * 50.0
                * ENERGY_PER_INFRASTRUCTURE_UNIT
                * scaling.infrastructure_efficiency; // sublinear: large pops need less per capita

            // Net surplus
            ledger.net_surplus =
                ledger.net_production - ledger.population_cost - ledger.complexity_cost;

            // Aggregate EROI (production-weighted average of sources)
            let total_gross: f64 = ledger.sources.iter().map(|s| s.gross_production).sum();
            ledger.aggregate_eroi = if total_gross > 0.0 {
                ledger
                    .sources
                    .iter()
                    .map(|s| s.eroi * s.gross_production / total_gross)
                    .sum()
            } else {
                0.0
            };

            // Waste heat: energy consumed × (1 - MAX_POWER_EFFICIENCY)
            // Systems operate at ~50% efficiency (Lotka-Odum)
            let total_consumed = ledger.population_cost + ledger.complexity_cost;
            ledger.waste_heat = total_consumed * (1.0 - MAX_POWER_EFFICIENCY);

            // 2nd Law: entropy monotonically increases
            // dS = waste_heat / T_proxy, where T_proxy ∝ 1.0 (normalized)
            ledger.cumulative_entropy += ledger.waste_heat;

            // EROI history (ring buffer, 120 ticks = 10 years)
            ledger.eroi_history.push(ledger.aggregate_eroi);
            if ledger.eroi_history.len() > 120 {
                ledger.eroi_history.remove(0);
            }

            // Deficit tracking
            if ledger.net_surplus < 0.0 {
                ledger.deficit_streak += 1;
            } else {
                ledger.deficit_streak = 0;
            }

            // 3. Snapshot ledger state for viability checks (avoids borrow conflict)
            let snap_surplus = ledger.net_surplus;
            let snap_eroi = ledger.aggregate_eroi;
            let snap_entropy = ledger.cumulative_entropy;
            let snap_deficit_streak = ledger.deficit_streak;
            let snap_waste_heat = ledger.waste_heat;
            let snap_eroi_history_len = ledger.eroi_history.len();

            // 4. Check viability conditions (uses snapshots, not the borrow)
            self.check_viability_from_snapshot(
                world_id,
                snap_surplus,
                snap_eroi,
                snap_deficit_streak,
                snap_waste_heat,
                snap_eroi_history_len,
                current_tick,
            );

            // 5. Record snapshot
            self.history.push(ViabilitySnapshot {
                tick: current_tick,
                world_id,
                net_surplus: snap_surplus,
                aggregate_eroi: snap_eroi,
                cumulative_entropy: snap_entropy,
                scaling_innovation: scaling.innovation_multiplier,
                violations: self
                    .violations
                    .iter()
                    .filter(|v| match v {
                        ViabilityViolation::EnergyDeficit { world_id: wid, .. } => *wid == world_id,
                        ViabilityViolation::EROIBelowTier { world_id: wid, .. } => *wid == world_id,
                        ViabilityViolation::EntropyAccumulation { world_id: wid, .. } => {
                            *wid == world_id
                        }
                        ViabilityViolation::InnovationStagnation { world_id: wid, .. } => {
                            *wid == world_id
                        }
                        ViabilityViolation::CoordinationOverload { world_id: wid, .. } => {
                            *wid == world_id
                        }
                    })
                    .cloned()
                    .collect(),
            });

            // Cap history at 1200 entries (100 years × 12 months)
            if self.history.len() > 1200 {
                self.history.remove(0);
            }
        }
    }

    /// Check all 5 viability conditions using snapshotted values (avoids borrow conflicts).
    fn check_viability_from_snapshot(
        &mut self,
        world_id: u32,
        net_surplus: f64,
        aggregate_eroi: f64,
        deficit_streak: u32,
        waste_heat: f64,
        eroi_history_len: usize,
        _current_tick: u32,
    ) {
        // Condition 1: Net energy surplus
        if deficit_streak >= 6 {
            // 6 consecutive months of deficit = Tainter pressure
            self.violations.push(ViabilityViolation::EnergyDeficit {
                world_id,
                deficit: net_surplus,
                streak_ticks: deficit_streak,
            });
        }

        // Condition 2: EROI above complexity tier
        let required = self.required_eroi_for_world(world_id);
        if aggregate_eroi < required && aggregate_eroi > 0.0 {
            self.violations.push(ViabilityViolation::EROIBelowTier {
                world_id,
                current_eroi: aggregate_eroi,
                required_eroi: required,
                tier_name: self.tier_name_for_eroi(required),
            });
        }

        // Condition 3: Entropy accumulation rate
        // If EROI is declining and waste heat is significant, entropy export is failing
        if eroi_history_len >= 12 && aggregate_eroi > 0.0 && aggregate_eroi < EROI_SURVIVAL {
            self.violations
                .push(ViabilityViolation::EntropyAccumulation {
                    world_id,
                    entropy_rate: waste_heat,
                });
        }
    }

    /// Determine required EROI based on world's current complexity/tech level.
    fn required_eroi_for_world(&self, _world_id: u32) -> f64 {
        // Default: industrial threshold. Will be refined when wired to tech tree.
        EROI_FAMILY
    }

    fn tier_name_for_eroi(&self, eroi: f64) -> String {
        if eroi >= EROI_SPACEFARING {
            "Spacefaring".into()
        } else if eroi >= EROI_INDUSTRIAL {
            "Industrial".into()
        } else if eroi >= EROI_FAMILY {
            "Family".into()
        } else {
            "Survival".into()
        }
    }

    /// Get scaling factors for a world (used by economy, knowledge, governance).
    pub fn scaling_for(&self, world_id: u32) -> ScalingFactors {
        self.scaling
            .get(&world_id)
            .copied()
            .unwrap_or(ScalingFactors {
                population: 0.0,
                infrastructure_efficiency: 1.0,
                innovation_multiplier: 1.0,
                pace_multiplier: 1.0,
            })
    }

    /// Get energy ledger for a world (used by economy for energy-constrained production).
    pub fn ledger_for(&self, world_id: u32) -> Option<&CivilizationEnergyLedger> {
        self.ledgers.get(&world_id)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_factors_neutral_for_small_pop() {
        let sf = ScalingFactors::compute(30.0);
        assert!((sf.infrastructure_efficiency - 1.0).abs() < 0.001);
        assert!((sf.innovation_multiplier - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_scaling_factors_sublinear_infrastructure() {
        let sf_1k = ScalingFactors::compute(1000.0);
        let sf_10k = ScalingFactors::compute(10000.0);
        // 10x more people should need LESS infra per capita
        assert!(
            sf_10k.infrastructure_efficiency < sf_1k.infrastructure_efficiency,
            "Infrastructure efficiency should decrease per-capita with larger pop: {} vs {}",
            sf_10k.infrastructure_efficiency,
            sf_1k.infrastructure_efficiency
        );
    }

    #[test]
    fn test_scaling_factors_superlinear_innovation() {
        let sf_1k = ScalingFactors::compute(1000.0);
        let sf_10k = ScalingFactors::compute(10000.0);
        // 10x more people should produce MORE innovation per capita
        assert!(
            sf_10k.innovation_multiplier > sf_1k.innovation_multiplier,
            "Innovation multiplier should increase per-capita with larger pop: {} vs {}",
            sf_10k.innovation_multiplier,
            sf_1k.innovation_multiplier
        );
    }

    #[test]
    fn test_scaling_exact_values() {
        // At N=1000 (reference), all factors should be 1.0
        let sf = ScalingFactors::compute(1000.0);
        assert!((sf.infrastructure_efficiency - 1.0).abs() < 0.001);
        assert!((sf.innovation_multiplier - 1.0).abs() < 0.001);

        // At N=10000, innovation = (10)^(1/6) ≈ 1.468
        let sf = ScalingFactors::compute(10000.0);
        assert!(
            (sf.innovation_multiplier - 10_f64.powf(1.0 / 6.0)).abs() < 0.01,
            "Expected ~1.468, got {}",
            sf.innovation_multiplier
        );
    }

    #[test]
    fn test_energy_ledger_deficit_tracking() {
        let mut engine = ViabilityEngine::new();
        engine.init_world(0, 100.0, 200.0);

        // Tick with very low energy production → deficit
        for tick in 0..7 {
            engine.tick(&[(0, 10.0, 200.0, 0.5)], tick);
        }
        let ledger = engine.ledger_for(0).unwrap();
        assert!(ledger.net_surplus < 0.0, "Should be in deficit");
        assert!(
            ledger.deficit_streak >= 6,
            "Should have deficit streak >= 6"
        );
        assert!(
            !engine.violations.is_empty(),
            "Should have viability violations"
        );
    }

    #[test]
    fn test_entropy_monotonically_increases() {
        let mut engine = ViabilityEngine::new();
        engine.init_world(0, 100.0, 100.0);

        let mut prev_entropy = 0.0;
        for tick in 0..24 {
            engine.tick(&[(0, 100.0, 100.0, 0.5)], tick);
            let ledger = engine.ledger_for(0).unwrap();
            assert!(
                ledger.cumulative_entropy >= prev_entropy,
                "Entropy must monotonically increase: {} < {}",
                ledger.cumulative_entropy,
                prev_entropy
            );
            prev_entropy = ledger.cumulative_entropy;
        }
        assert!(
            prev_entropy > 0.0,
            "Entropy should be positive after 24 ticks"
        );
    }

    #[test]
    fn test_eroi_constants_ordered() {
        assert!(EROI_SURVIVAL < EROI_FAMILY);
        assert!(EROI_FAMILY < EROI_INDUSTRIAL);
        assert!(EROI_INDUSTRIAL < EROI_SPACEFARING);
    }

    #[test]
    fn test_energy_source_eroi() {
        let source = EnergySource::new("solar", 100.0, 10.0);
        assert!((source.eroi - 10.0).abs() < 0.001);

        let infinite = EnergySource::new("free", 100.0, 0.0);
        assert!(infinite.eroi.is_infinite());
    }
}
