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

fn default_true() -> bool {
    true
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
    /// Enable trust-weighted governance (tier-based decision authority).
    /// Trust-weighted governance: voting power and authority scale with
    /// trust depth (4D profile: identity, reputation, community, engagement).
    /// When false, all agents vote equally regardless of trust — the control
    /// condition. A/B test showed +6.9% CVS improvement with trust weighting.
    pub trust_weighted_governance: bool,
    pub faction_enabled: bool,
    pub amendment_enabled: bool,
    pub outer_system_fission_enabled: bool,
    pub speciation_friction_enabled: bool,
    pub hostile_guardian: bool,
    /// Enable the probabilistic disaster engine (7 categories, 40 event types).
    /// Default true. Set false for A/B comparison (disasters OFF).
    pub disasters_enabled: bool,
    /// Enable hybrid Earth model: 12 aggregate regions + Spaceport Funnel.
    /// When true, Earth is modeled as aggregate demographics (no individual agents)
    /// and off-world colonists are instantiated via the Spaceport Funnel.
    /// Default false for backward compatibility.
    pub hybrid_earth: bool,

    // === CANONICAL GOVERNANCE SYSTEMS ===
    /// Metabolism cycle configuration (4-phase governance rhythm).
    #[serde(default)]
    pub metabolism: crate::metabolism::MetabolismConfig,

    /// Governance hardening (constitutional anti-tyranny invariants).
    #[serde(default)]
    pub governance_hardening: crate::governance_hardening::GovernanceHardeningConfig,

    // === REALISM TOGGLES — scientific instrument controls ===
    /// Turchin secular cycles: elite overproduction → crisis → civil war.
    /// When false, no Turchin dynamics — society is conflict-free (utopia).
    #[serde(default = "default_true")]
    pub turchin_cycles_enabled: bool,
    /// FEP-coupled immiseration: prediction error drives suffering thermodynamically.
    /// When false, immiseration uses abstract Gini only (standard Turchin).
    #[serde(default = "default_true")]
    pub fep_immiseration_enabled: bool,
    /// Sacred Stillness: rest as existential infrastructure (6× rebuild multiplier).
    /// When false, Stillness harmony has no effect on recovery rate.
    #[serde(default = "default_true")]
    pub sacred_stillness_enabled: bool,
    /// Stoichiometric mass ledger for closed-loop ECLSS (generation ships only).
    #[serde(default)]
    pub stoichiometry_enabled: bool,
    /// Cosmic epistemic decay: knowledge degrades from radiation + isolation.
    #[serde(default)]
    pub epistemic_decay_enabled: bool,
    /// Age-structured demographics with Leslie matrix and dependency ratios.
    #[serde(default)]
    pub age_demographics_enabled: bool,

    /// A2 counterfactual: enable the Phase 2 governance machinery (8D
    /// sovereign profile, restorative justice, Mycelix rate limits).
    /// Default true. Set false to run the scalar-Phi + MYCEL baseline
    /// against the same attackers, isolating the defense's contribution.
    #[serde(default = "default_true")]
    pub phase2_enabled: bool,

    // === DISASTER CATEGORY TOGGLES ===
    /// Solar weather events (M-class, X-class, Carrington, SPE).
    #[serde(default = "default_true")]
    pub disasters_solar_enabled: bool,
    /// Impact events (micrometeorite, meteorite).
    #[serde(default = "default_true")]
    pub disasters_impact_enabled: bool,
    /// Planetary environment (dust storms, moonquakes, tides).
    #[serde(default = "default_true")]
    pub disasters_planetary_enabled: bool,
    /// ECLSS failures (O2, water, CO2, thermal, power, hydroponics).
    #[serde(default = "default_true")]
    pub disasters_eclss_enabled: bool,
    /// Psychological events (winter-over, conflict, cognitive impairment).
    #[serde(default = "default_true")]
    pub disasters_psychological_enabled: bool,
    /// Technology events (breakthroughs, regressions).
    #[serde(default = "default_true")]
    pub disasters_technology_enabled: bool,
    /// Civilization dynamics (Tainter collapse, Turchin crises, Kessler).
    #[serde(default = "default_true")]
    pub disasters_civilization_enabled: bool,

    // === C: SCENARIO MODE — player/researcher policy choices ===
    /// Colony project priority strategy.
    pub project_strategy: ProjectStrategy,
    /// Birth policy (affects birth rate modifier).
    pub birth_policy: BirthPolicy,
    /// Resource allocation priority (affects what fraction goes to which sector).
    pub resource_priority: ResourcePriority,
    /// Trade openness (0.0 = autarky, 1.0 = free trade).
    pub trade_openness: f64,
    /// Defense spending fraction (0.0 - 0.3). Diverts labor from production.
    pub defense_spending: f64,
    /// Exploration investment fraction (0.0 - 0.2). Accelerates discovery.
    pub exploration_investment: f64,

    // === D: COORDINATION SCIENCE EXPERIMENT CONTROLS ===
    /// Initial coordination science understanding for all agents [0.0, 1.0].
    /// 0.0 = no coordination understanding (baseline).
    /// 0.5 = moderate (coordination-literate condition).
    /// Orthogonal to education_level and consciousness.
    #[serde(default)]
    pub coordination_understanding_initial: f64,
    /// Education level boost applied at initialization (for education-only control arm).
    /// Added on top of default education_level.
    #[serde(default)]
    pub education_boost: f64,
    /// Initial consciousness level boost [0.0, 1.0].
    /// When > 0, agents start with elevated consciousness state
    /// (level, meta_awareness, coherence all boosted).
    #[serde(default)]
    pub consciousness_boost: f64,
}

/// How the colony prioritizes construction projects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectStrategy {
    /// AI governor makes optimal decisions (default).
    Balanced,
    /// Prioritize survival infrastructure (ECLSS, food, medical).
    SurvivalFirst,
    /// Prioritize growth (habitats, greenhouses, water).
    GrowthFirst,
    /// Prioritize science and exploration (vehicles, comms).
    ScienceFirst,
    /// Prioritize self-sufficiency (fabrication, power, water).
    IndependenceFirst,
}

/// Birth rate policy for colony governance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BirthPolicy {
    /// Natural birth rate, no intervention.
    Natural,
    /// Pro-natal: +50% birth rate. Baby boom → future labor, but childcare burden now.
    ProNatal,
    /// Population control: -50% birth rate. Reduces resource pressure, but aging colony.
    PopulationControl,
    /// Replacement only: birth rate ≈ death rate.
    ReplacementOnly,
}

/// Resource allocation priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourcePriority {
    /// Equal allocation across all sectors.
    Balanced,
    /// Heavy investment in engineering + manufacturing.
    Industrial,
    /// Heavy investment in agriculture + medicine.
    Biological,
    /// Heavy investment in science + education.
    Knowledge,
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
            trust_weighted_governance: true,
            faction_enabled: true,
            amendment_enabled: true,
            outer_system_fission_enabled: true,
            speciation_friction_enabled: true,
            hostile_guardian: false,
            disasters_enabled: true,
            hybrid_earth: false,
            metabolism: crate::metabolism::MetabolismConfig::default(),
            governance_hardening: crate::governance_hardening::GovernanceHardeningConfig::default(),
            turchin_cycles_enabled: true,
            fep_immiseration_enabled: true,
            sacred_stillness_enabled: true,
            stoichiometry_enabled: false,
            epistemic_decay_enabled: false,
            age_demographics_enabled: false,
            disasters_solar_enabled: true,
            disasters_impact_enabled: true,
            disasters_planetary_enabled: true,
            disasters_eclss_enabled: true,
            disasters_psychological_enabled: true,
            disasters_technology_enabled: true,
            disasters_civilization_enabled: true,
            project_strategy: ProjectStrategy::Balanced,
            birth_policy: BirthPolicy::Natural,
            resource_priority: ResourcePriority::Balanced,
            trade_openness: 0.8,
            defense_spending: 0.05,
            exploration_investment: 0.05,
            coordination_understanding_initial: 0.0,
            education_boost: 0.0,
            consciousness_boost: 0.0,
            phase2_enabled: true,
        }
    }
}

impl PolicyConfig {
    /// Legacy mode: standard human governance WITHOUT trust weighting or FEP.
    /// This is the control group for proving Symthaea/Mycelix are necessary.
    pub fn legacy_mode() -> Self {
        Self {
            trust_weighted_governance: false,
            fep_immiseration_enabled: false,
            sacred_stillness_enabled: false,
            turchin_cycles_enabled: true, // Keep conflict — it's what breaks legacy
            faction_enabled: true,
            ..Self::default()
        }
    }

    /// Utopia mode: all disasters and conflict disabled. The ceiling test.
    pub fn utopia_mode() -> Self {
        Self {
            disasters_enabled: false,
            turchin_cycles_enabled: false,
            faction_enabled: false,
            ..Self::default()
        }
    }

    /// Create a "Survival First" preset (conservative, risk-averse).
    pub fn survival_first() -> Self {
        Self {
            project_strategy: ProjectStrategy::SurvivalFirst,
            birth_policy: BirthPolicy::ReplacementOnly,
            resource_priority: ResourcePriority::Biological,
            defense_spending: 0.1,
            exploration_investment: 0.02,
            trade_openness: 0.5,
            ..Self::default()
        }
    }

    /// Create a "Growth First" preset (aggressive expansion).
    pub fn growth_first() -> Self {
        Self {
            project_strategy: ProjectStrategy::GrowthFirst,
            birth_policy: BirthPolicy::ProNatal,
            resource_priority: ResourcePriority::Balanced,
            defense_spending: 0.02,
            exploration_investment: 0.05,
            trade_openness: 0.9,
            ..Self::default()
        }
    }

    /// Create a "Science First" preset (knowledge economy).
    pub fn science_first() -> Self {
        Self {
            project_strategy: ProjectStrategy::ScienceFirst,
            birth_policy: BirthPolicy::Natural,
            resource_priority: ResourcePriority::Knowledge,
            defense_spending: 0.02,
            exploration_investment: 0.15,
            trade_openness: 1.0,
            ..Self::default()
        }
    }

    /// Create an "Independence First" preset (autarky-oriented).
    pub fn independence_first() -> Self {
        Self {
            project_strategy: ProjectStrategy::IndependenceFirst,
            birth_policy: BirthPolicy::ProNatal,
            resource_priority: ResourcePriority::Industrial,
            defense_spending: 0.15,
            exploration_investment: 0.05,
            trade_openness: 0.3,
            ..Self::default()
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

    /// 300-year configuration (3600 ticks).
    pub fn default_300_year() -> Self {
        let mut cfg = Self::default_150_year();
        cfg.total_ticks = 300 * TICKS_PER_YEAR;
        cfg.epoch_configs.push(EpochConfig {
            id: 7,
            name: "Convergence".into(),
            start_tick: 150 * TICKS_PER_YEAR,
            end_tick: 200 * TICKS_PER_YEAR,
            population_trigger: None,
            self_sufficiency_trigger: None,
        });
        cfg.epoch_configs.push(EpochConfig {
            id: 8,
            name: "Secondary Growth".into(),
            start_tick: 200 * TICKS_PER_YEAR,
            end_tick: 250 * TICKS_PER_YEAR,
            population_trigger: None,
            self_sufficiency_trigger: None,
        });
        cfg.epoch_configs.push(EpochConfig {
            id: 9,
            name: "Legacy".into(),
            start_tick: 250 * TICKS_PER_YEAR,
            end_tick: 300 * TICKS_PER_YEAR,
            population_trigger: None,
            self_sufficiency_trigger: None,
        });
        cfg
    }

    /// 777-year configuration (9324 ticks).
    pub fn default_777_year() -> Self {
        let mut cfg = Self::default_300_year();
        cfg.total_ticks = 777 * TICKS_PER_YEAR;
        cfg.initial_worlds.push(WorldSeedConfig {
            name: "Europa Station".into(),
            location: "Europa".into(),
            founding_tick: 2400,
            initial_population: 200,
            initial_resources: 0.15,
        });
        cfg.initial_worlds.push(WorldSeedConfig {
            name: "Titan Outpost".into(),
            location: "Titan".into(),
            founding_tick: 3600,
            initial_population: 150,
            initial_resources: 0.1,
        });
        cfg.epoch_configs.push(EpochConfig {
            id: 10,
            name: "Millennium".into(),
            start_tick: 300 * TICKS_PER_YEAR,
            end_tick: 777 * TICKS_PER_YEAR,
            population_trigger: None,
            self_sufficiency_trigger: None,
        });
        cfg
    }

    /// 777-year configuration with Resontia Earth-hardening enabled.
    ///
    /// Same as `default_777_year()` but includes Resontia infrastructure
    /// on Earth starting at year 50 (Epoch 3: Maturation).
    /// Use with `crate::resontia::default_resontia_config()` for the
    /// Resontia-specific parameters.
    pub fn default_resontia_777_year() -> Self {
        // Base 777-year config — Resontia config is separate (ResontiaConfig struct)
        Self::default_777_year()
    }

    /// 777-year configuration with hybrid Earth model (12 aggregate regions
    /// + Spaceport Funnel for macro-to-micro agent instantiation).
    ///
    /// Earth is modeled as 12 regional populations using aggregate demographics
    /// (no individual Earth agents). Off-world colonists are instantiated from
    /// regional statistics via the Spaceport Funnel when launched.
    pub fn default_hybrid_777_year() -> Self {
        let mut cfg = Self::default_777_year();
        cfg.policy.hybrid_earth = true;
        cfg
    }

    /// 1000-year configuration (12000 ticks).
    pub fn default_1000_year() -> Self {
        let mut cfg = Self::default_300_year();
        cfg.total_ticks = 1000 * TICKS_PER_YEAR;
        cfg.initial_worlds.push(WorldSeedConfig {
            name: "Europa Station".into(),
            location: "Europa".into(),
            founding_tick: 2400,
            initial_population: 200,
            initial_resources: 0.15,
        });
        cfg.initial_worlds.push(WorldSeedConfig {
            name: "Titan Outpost".into(),
            location: "Titan".into(),
            founding_tick: 3600,
            initial_population: 150,
            initial_resources: 0.1,
        });
        cfg.epoch_configs.push(EpochConfig {
            id: 10,
            name: "Millennium".into(),
            start_tick: 300 * TICKS_PER_YEAR,
            end_tick: 1000 * TICKS_PER_YEAR,
            population_trigger: None,
            self_sufficiency_trigger: None,
        });
        cfg
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
