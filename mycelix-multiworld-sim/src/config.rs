// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Simulation configuration: epochs, world seeds, and timing constants.

use serde::{Deserialize, Serialize};

/// Ticks per simulated year (monthly resolution).
pub const TICKS_PER_YEAR: u32 = 12;

/// Ticks per generation (~25 years).
pub const GENERATION_TICKS: u32 = 300;

/// Unique epoch identifier.
pub type EpochId = u8;

/// Top-level simulation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Total simulation ticks (1800 = 150 years at monthly resolution).
    pub total_ticks: u32,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Initial world definitions.
    pub initial_worlds: Vec<WorldSeedConfig>,
    /// Epoch definitions (progression milestones).
    pub epoch_configs: Vec<EpochConfig>,
    /// Policy knobs for scenario comparison.
    pub policy: PolicyConfig,
}

/// Configurable policy knobs for A/B scenario testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    /// Monthly pair-bonding probability (default 0.02). Set to 0.0 to disable.
    pub pair_bond_rate: f64,
    /// Care worker load reduction multiplier (default 1.0). Set to 0.0 for no care economy.
    pub care_effectiveness: f64,
    /// Astropharmacy consciousness boost (default 0.3). Set to 0.0 to disable.
    pub pharma_boost: f64,
    /// Deep-space social decay multiplier (default 1.5). Set to 1.0 to disable.
    pub deep_space_isolation_mult: f64,
    /// Inter-world migration enabled (default true).
    pub migration_enabled: bool,
    /// Migration rate: max migrants per colony per 6-tick cycle (default 3).
    pub migration_max_per_cycle: u32,
    /// Enable the education guild tick (peer teaching, epistemic foraging).
    /// Default true. Set false for A/B comparison.
    pub education_enabled: bool,
    /// Enable consciousness-gated governance (tier-based decision authority).
    /// Default true. When false, all agents are treated as Observer tier —
    /// no consciousness-based filtering of decisions. This is the control
    /// condition for proving that consciousness gating improves outcomes.
    pub consciousness_gating_enabled: bool,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            pair_bond_rate: 0.02,
            care_effectiveness: 1.0,
            pharma_boost: 0.3,
            deep_space_isolation_mult: 1.5,
            migration_enabled: true,
            migration_max_per_cycle: 3,
            education_enabled: true,
            consciousness_gating_enabled: true,
        }
    }
}

/// Seed configuration for founding a world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSeedConfig {
    pub name: String,
    /// Location label: "Earth", "Moon", "Mars", "Europa", etc.
    pub location: String,
    /// Tick at which this world is founded (0 for starting worlds).
    pub founding_tick: u32,
    /// Number of initial colonists.
    pub initial_population: usize,
    /// Starting resource multiplier (1.0 = baseline Earth-equivalent).
    pub initial_resources: f64,
}

/// Epoch milestone configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochConfig {
    pub id: EpochId,
    pub name: String,
    /// Tick at which this epoch begins (if time-triggered).
    pub start_tick: u32,
    /// Tick at which this epoch ends.
    pub end_tick: u32,
    /// Population threshold that can trigger epoch transition early.
    pub population_trigger: Option<usize>,
    /// Self-sufficiency ratio (0.0-1.0) that can trigger epoch transition.
    pub self_sufficiency_trigger: Option<f64>,
}

impl SimulationConfig {
    /// Canonical 150-year, 7-epoch configuration.
    ///
    /// Epochs:
    /// 0: Foundation (years 0-10) — Moon base established
    /// 1: Survival (years 10-25) — self-sufficiency push
    /// 2: Growth (years 25-50) — population expansion, Mars colony
    /// 3: Maturation (years 50-75) — cultural divergence, governance evolution
    /// 4: Expansion (years 75-100) — outer system colonies
    /// 5: Integration (years 100-125) — interworld governance
    /// 6: Transcendence (years 125-150) — consciousness civilization
    pub fn default_150_year() -> Self {
        Self {
            policy: PolicyConfig::default(),
            total_ticks: 150 * TICKS_PER_YEAR,
            seed: 42,
            initial_worlds: vec![
                WorldSeedConfig {
                    name: "Earth".into(),
                    location: "Earth".into(),
                    founding_tick: 0,
                    // Representative sample — Earth is the supply depot, not the focus.
                    // Off-world colonies are where agent-level dynamics matter.
                    initial_population: 500,
                    initial_resources: 1.0,
                },
                WorldSeedConfig {
                    name: "Artemis Base".into(),
                    location: "Moon".into(),
                    founding_tick: 0,
                    initial_population: 12,
                    initial_resources: 0.3,
                },
                WorldSeedConfig {
                    name: "Ares Colony".into(),
                    location: "Mars".into(),
                    founding_tick: 25 * TICKS_PER_YEAR,
                    initial_population: 50,
                    initial_resources: 0.2,
                },
            ],
            epoch_configs: vec![
                EpochConfig {
                    id: 0,
                    name: "Foundation".into(),
                    start_tick: 0,
                    end_tick: 10 * TICKS_PER_YEAR,
                    population_trigger: None,
                    self_sufficiency_trigger: None,
                },
                EpochConfig {
                    id: 1,
                    name: "Survival".into(),
                    start_tick: 10 * TICKS_PER_YEAR,
                    end_tick: 25 * TICKS_PER_YEAR,
                    population_trigger: Some(100),
                    self_sufficiency_trigger: Some(0.5),
                },
                EpochConfig {
                    id: 2,
                    name: "Growth".into(),
                    start_tick: 25 * TICKS_PER_YEAR,
                    end_tick: 50 * TICKS_PER_YEAR,
                    population_trigger: Some(1_000),
                    self_sufficiency_trigger: Some(0.7),
                },
                EpochConfig {
                    id: 3,
                    name: "Maturation".into(),
                    start_tick: 50 * TICKS_PER_YEAR,
                    end_tick: 75 * TICKS_PER_YEAR,
                    population_trigger: Some(10_000),
                    self_sufficiency_trigger: Some(0.85),
                },
                EpochConfig {
                    id: 4,
                    name: "Expansion".into(),
                    start_tick: 75 * TICKS_PER_YEAR,
                    end_tick: 100 * TICKS_PER_YEAR,
                    population_trigger: Some(50_000),
                    self_sufficiency_trigger: Some(0.9),
                },
                EpochConfig {
                    id: 5,
                    name: "Integration".into(),
                    start_tick: 100 * TICKS_PER_YEAR,
                    end_tick: 125 * TICKS_PER_YEAR,
                    population_trigger: None,
                    self_sufficiency_trigger: Some(0.95),
                },
                EpochConfig {
                    id: 6,
                    name: "Transcendence".into(),
                    start_tick: 125 * TICKS_PER_YEAR,
                    end_tick: 150 * TICKS_PER_YEAR,
                    population_trigger: None,
                    self_sufficiency_trigger: None,
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_creates_valid_7_epoch_structure() {
        let cfg = SimulationConfig::default_150_year();
        assert_eq!(cfg.total_ticks, 1800);
        assert_eq!(cfg.epoch_configs.len(), 7);
        assert_eq!(cfg.initial_worlds.len(), 3);

        // Epochs cover the full range
        assert_eq!(cfg.epoch_configs[0].start_tick, 0);
        assert_eq!(cfg.epoch_configs[6].end_tick, 1800);

        // Each epoch id matches its index
        for (i, e) in cfg.epoch_configs.iter().enumerate() {
            assert_eq!(e.id as usize, i);
        }

        // Mars founded at year 25
        assert_eq!(cfg.initial_worlds[2].founding_tick, 300);
    }
}
