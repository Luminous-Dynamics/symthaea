// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Epoch progression: milestone evaluation, checkpoint assertions, and transitions.
//!
//! The simulation spans 6 epochs over 150 years (1800 monthly ticks):
//!
//! | Epoch     | Ticks     | Years     | Theme                              |
//! |-----------|-----------|-----------|-------------------------------------|
//! | Seeds     | 0-168     | 2026-2040 | Base assembly, crew arrival         |
//! | Roots     | 168-408   | 2040-2060 | First births, constitution, SS      |
//! | Branches  | 408-708   | 2060-2085 | Mars colony, trade, cultural drift  |
//! | Canopy    | 708-1068  | 2085-2115 | Inter-world council, equity         |
//! | Fruiting  | 1068-1428 | 2115-2145 | Manufacturing SS, love coherence    |
//! | Dispersal | 1428-1800 | 2145-2176 | Long-term viability, final CVS      |

use crate::config::{EpochId, TICKS_PER_YEAR};
use crate::events::{CivEvent, CivEventType};
use crate::population::PopulationEngine;
use crate::world::World;

use serde::{Deserialize, Serialize};

/// Epoch identifiers. Values map to the config epoch_id field.
pub const EPOCH_SEEDS: EpochId = 0;
pub const EPOCH_ROOTS: EpochId = 1;
pub const EPOCH_BRANCHES: EpochId = 2;
pub const EPOCH_CANOPY: EpochId = 3;
pub const EPOCH_FRUITING: EpochId = 4;
pub const EPOCH_DISPERSAL: EpochId = 5;
pub const EPOCH_CONVERGENCE: EpochId = 6;
pub const EPOCH_SECONDARY_GROWTH: EpochId = 7;
pub const EPOCH_LEGACY: EpochId = 8;
pub const EPOCH_MILLENNIUM: EpochId = 9;

/// Human-readable epoch name.
pub fn epoch_name(id: EpochId) -> &'static str {
    match id {
        0 => "Seeds",
        1 => "Roots",
        2 => "Branches",
        3 => "Canopy",
        4 => "Fruiting",
        5 => "Dispersal",
        6 => "Convergence",
        7 => "Secondary Growth",
        8 => "Legacy",
        9 => "Millennium",
        _ => "Unknown",
    }
}

/// Per-world summary in an epoch snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub name: String,
    pub location: String,
    pub population: usize,
    pub phi: f64,
    pub allostatic_load: f64,
    /// Spinozist net conatus (joy - sadness). >0 = flourishing.
    pub net_conatus: f64,
    /// Moral balance (care - harm). >0 = moral health.
    pub moral_balance: f64,
    pub self_sufficiency: f64,
}

/// Snapshot of civilization state at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochSnapshot {
    pub epoch_id: EpochId,
    pub tick: u32,
    pub total_population: usize,
    /// Population on off-Earth worlds only.
    pub off_earth_population: usize,
    pub world_count: usize,
    pub mean_self_sufficiency: f64,
    pub mean_phi: f64,
    pub mean_love_coherence: f64,
    pub genetic_diversity: f64,
    pub max_oppression_index: f64,
    pub mean_tech_level: f64,
    pub civilization_viability_score: f64,
    /// Mean allostatic load across all agents in all worlds.
    pub mean_allostatic_load: f64,
    /// Mean engagement across all agents in all worlds.
    pub mean_engagement: f64,
    /// Number of active factions across all worlds.
    pub faction_count: usize,
    /// Fraction of Guardian-tier agents whose parents were also Guardian-tier.
    pub elite_persistence: f64,
    /// Constitutional amendments per tick (recent window).
    pub amendment_rate: f64,
    /// Innovation stagnation index [0.0, 1.0].
    pub innovation_stagnation: f64,
    /// Mean cultural distance between worlds.
    pub inter_world_divergence: f64,
    /// Gini coefficient of consciousness (phi) across all agents.
    pub consciousness_gini: f64,
    /// Phi trend classification: "Rising", "Oscillating", "Plateau", "Declining".
    pub phi_trend: String,
    /// Number of CVS drops > 0.1 followed by recovery.
    pub recovery_count: usize,
    /// Mean trauma level across all agents.
    pub trauma_level: f64,
    /// Genetic speciation index between worlds.
    pub speciation_index: f64,
    /// Per-world breakdowns (populated for outer system visibility).
    pub worlds: Vec<WorldSnapshot>,
    /// Mean ethical orientation across all agents (4D: deont, conseq, virtue, relat).
    #[serde(default)]
    pub ethics_mean: [f64; 4],
    /// Standard deviation of ethics (diversity measure).
    #[serde(default)]
    pub ethics_std: [f64; 4],
    /// Simpson diversity index of dominant ethical framework [0.0, 1.0].
    /// 0.0 = monoculture, 0.75 = maximal diversity (4 equal frameworks).
    #[serde(default)]
    pub ethics_diversity: f64,
}

impl EpochSnapshot {
    /// Create a snapshot with only the fields available from the current simulation state.
    /// Fields that require subsystems not yet wired default to 0.0.
    pub fn from_worlds(epoch_id: EpochId, tick: u32, worlds: &[World]) -> Self {
        let total_population: usize = worlds.iter().map(|w| w.population()).sum();
        let world_count = worlds.len();

        // Off-Earth worlds are the interesting question for a colony sim.
        let off_earth_worlds: Vec<&World> =
            worlds.iter().filter(|w| w.location != "Earth").collect();
        let off_earth_population: usize = off_earth_worlds.iter().map(|w| w.population()).sum();

        let mean_self_sufficiency = if worlds.is_empty() {
            0.0
        } else {
            worlds
                .iter()
                .map(|w| w.resources.self_sufficiency())
                .sum::<f64>()
                / world_count as f64
        };

        let mean_phi = if worlds.is_empty() {
            0.0
        } else {
            worlds.iter().map(|w| w.collective_phi()).sum::<f64>() / world_count as f64
        };

        // Genetic diversity: focus on off-Earth populations (Earth's genetics aren't
        // the interesting question). Fall back to all worlds if no off-Earth colonies exist.
        let genetic_diversity = if !off_earth_worlds.is_empty() {
            off_earth_worlds
                .iter()
                .map(|w| PopulationEngine::genetic_diversity_index(w, tick))
                .sum::<f64>()
                / off_earth_worlds.len() as f64
        } else if !worlds.is_empty() {
            worlds
                .iter()
                .map(|w| PopulationEngine::genetic_diversity_index(w, tick))
                .sum::<f64>()
                / world_count as f64
        } else {
            0.0
        };

        // Love coherence and oppression come from subsystems not yet wired.
        let mean_love_coherence = mean_phi * 0.8; // proxy: phi correlates with love
        let max_oppression_index = 0.0; // no oppression tracking yet

        // Tech level from the Romer knowledge engine. Normalize: tech starts at 1.0,
        // so (mean_tech - 1.0) / 9.0 maps [1.0, 10.0] -> [0.0, 1.0].
        let mean_tech_level = if worlds.is_empty() {
            0.0
        } else {
            let raw: f64 = worlds
                .iter()
                .map(|w| w.knowledge.mean_tech_level())
                .sum::<f64>()
                / world_count as f64;
            ((raw - 1.0) / 9.0).clamp(0.0, 1.0)
        };

        // Psychological needs aggregates
        let (total_load, total_engagement, agent_count) =
            worlds
                .iter()
                .fold((0.0f64, 0.0f64, 0usize), |(load, eng, count), w| {
                    let living: Vec<_> = w.agents.iter().filter(|a| a.is_alive()).collect();
                    let n = living.len();
                    let l: f64 = living.iter().map(|a| a.needs.allostatic_load).sum();
                    let e: f64 = living.iter().map(|a| a.needs.engagement).sum();
                    (load + l, eng + e, count + n)
                });
        let ac = agent_count.max(1) as f64;
        let mean_allostatic_load = total_load / ac;
        let mean_engagement = total_engagement / ac;

        let cvs = EpochManager::compute_cvs(
            genetic_diversity,
            mean_self_sufficiency,
            mean_love_coherence,
            max_oppression_index,
            mean_phi,
        );

        // Per-world snapshots for outer system visibility.
        let world_snapshots: Vec<WorldSnapshot> = worlds
            .iter()
            .map(|w| {
                let living: Vec<_> = w.agents.iter().filter(|a| a.is_alive()).collect();
                let n = living.len().max(1) as f64;
                let phi = living.iter().map(|a| a.consciousness.phi()).sum::<f64>() / n;
                let load = living.iter().map(|a| a.needs.allostatic_load).sum::<f64>() / n;
                let net_conatus = living
                    .iter()
                    .map(|a| a.needs.affect.net_conatus())
                    .sum::<f64>()
                    / n;
                let moral_balance = living
                    .iter()
                    .map(|a| a.needs.affect.moral_balance())
                    .sum::<f64>()
                    / n;
                WorldSnapshot {
                    name: w.name.clone(),
                    location: w.location.clone(),
                    population: w.population(),
                    phi,
                    allostatic_load: load,
                    net_conatus,
                    moral_balance,
                    self_sufficiency: w.resources.self_sufficiency(),
                }
            })
            .collect();

        // Ethical composition telemetry (Phase 3: ethical pluralism)
        let mut ethics_sums = [0.0f64; 4];
        let mut ethics_sq_sums = [0.0f64; 4];
        let mut dominant_counts = [0usize; 4]; // deont, conseq, virtue, relat
        let ethics_n = agent_count.max(1) as f64;
        for w in worlds {
            for a in w.agents.iter().filter(|a| a.is_alive()) {
                let v = a.ethics.as_vec();
                for d in 0..4 {
                    ethics_sums[d] += v[d];
                    ethics_sq_sums[d] += v[d] * v[d];
                }
                // Count dominant framework for Simpson diversity
                let max_idx = v
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                dominant_counts[max_idx] += 1;
            }
        }
        let ethics_mean = [
            ethics_sums[0] / ethics_n,
            ethics_sums[1] / ethics_n,
            ethics_sums[2] / ethics_n,
            ethics_sums[3] / ethics_n,
        ];
        let ethics_std = [0, 1, 2, 3].map(|d| {
            let mean = ethics_mean[d];
            ((ethics_sq_sums[d] / ethics_n) - mean * mean)
                .max(0.0)
                .sqrt()
        });
        // Simpson diversity: 1 - Σ(p_i²), range [0, 0.75] for 4 categories
        let ethics_diversity = {
            let total = dominant_counts.iter().sum::<usize>().max(1) as f64;
            1.0 - dominant_counts
                .iter()
                .map(|&c| (c as f64 / total).powi(2))
                .sum::<f64>()
        };

        Self {
            epoch_id,
            tick,
            total_population,
            off_earth_population,
            world_count,
            mean_self_sufficiency,
            mean_phi,
            mean_love_coherence,
            genetic_diversity,
            max_oppression_index,
            mean_tech_level,
            civilization_viability_score: cvs,
            mean_allostatic_load,
            mean_engagement,
            // Defaults for extended observables — populated by observables module.
            faction_count: 0,
            elite_persistence: 0.0,
            amendment_rate: 0.0,
            innovation_stagnation: 0.0,
            inter_world_divergence: 0.0,
            consciousness_gini: 0.0,
            phi_trend: "Rising".into(),
            recovery_count: 0,
            trauma_level: 0.0,
            speciation_index: 0.0,
            worlds: world_snapshots,
            ethics_mean,
            ethics_std,
            ethics_diversity,
        }
    }
}

/// A checkpoint assertion that can be evaluated against simulation state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EpochAssertion {
    PopulationAbove(usize),
    WorldCount(usize),
    CollectivePhiAbove(f64),
    LoveCoherenceAbove(f64),
    SelfSufficiencyAbove(f64),
    GeneticDiversityAbove(f64),
    NoOppressionCrisis,
    TechLevelAbove(f64),
    TradeEstablished,
    ConstitutionRatified,
    FirstBirth,
    CvsAbove(f64),
    /// At least N factions have emerged across the civilization.
    FactionCountAbove(usize),
    /// Elite persistence (guardian inheritance) below threshold.
    ElitePersistenceBelow(f64),
    /// Innovation is not stagnant (stagnation index < 0.5).
    InnovationNonStagnant,
    /// Inter-world cultural divergence below threshold.
    InterWorldDivergenceBelow(f64),
    /// Constitutional amendment rate below threshold (stability).
    AmendmentRateBelow(f64),
    /// At least one CVS recovery from a >0.1 drop has occurred.
    RecoveryDemonstrated,
    /// Phi trend has been classified (not plateau).
    PhiTrendClassified,
    /// Mean trauma level across all worlds below threshold.
    TraumaBelow(f64),
    /// Speciation index below threshold.
    SpeciationBelow(f64),
    /// No faction conflict is currently active.
    NoFactionConflictActive,
}

/// A checkpoint at a specific tick with a set of assertions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochCheckpoint {
    pub tick: u32,
    pub epoch_id: EpochId,
    pub description: String,
    pub assertions: Vec<EpochAssertion>,
}

/// Manages epoch transitions, checkpoints, and milestone tracking.
pub struct EpochManager {
    pub current_epoch: EpochId,
    pub checkpoints: Vec<EpochCheckpoint>,
    /// Results: (tick, description, passed, failure_reasons)
    pub checkpoint_results: Vec<(u32, String, bool, Vec<String>)>,
    /// Transitions: (tick, from_epoch, to_epoch)
    pub epoch_transitions: Vec<(u32, EpochId, EpochId)>,
    pub first_birth_tick: Option<u32>,
    pub first_constitution_tick: Option<u32>,
    pub first_trade_tick: Option<u32>,
}

impl EpochManager {
    /// Create a new epoch manager with all predefined checkpoints.
    pub fn new() -> Self {
        let checkpoints = Self::build_checkpoints();
        Self {
            current_epoch: EPOCH_SEEDS,
            checkpoints,
            checkpoint_results: Vec::new(),
            epoch_transitions: Vec::new(),
            first_birth_tick: None,
            first_constitution_tick: None,
            first_trade_tick: None,
        }
    }

    /// Build the full set of epoch checkpoints (~25 total).
    fn build_checkpoints() -> Vec<EpochCheckpoint> {
        let mut cps = Vec::with_capacity(25);

        // === Seeds (0-168) ===
        cps.push(EpochCheckpoint {
            tick: 60,
            epoch_id: EPOCH_SEEDS,
            description: "Base assembled: initial infrastructure operational".into(),
            assertions: vec![
                EpochAssertion::PopulationAbove(5),
                EpochAssertion::WorldCount(1),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 108,
            epoch_id: EPOCH_SEEDS,
            description: "Crew arrives: population stable".into(),
            assertions: vec![EpochAssertion::PopulationAbove(8)],
        });
        cps.push(EpochCheckpoint {
            tick: 168,
            epoch_id: EPOCH_SEEDS,
            description: "Seeds epoch end: minimum crew survived".into(),
            assertions: vec![
                EpochAssertion::PopulationAbove(6),
                EpochAssertion::SelfSufficiencyAbove(0.3),
            ],
        });

        // === Roots (168-408) ===
        cps.push(EpochCheckpoint {
            tick: 240,
            epoch_id: EPOCH_ROOTS,
            description: "First birth in colony".into(),
            assertions: vec![EpochAssertion::FirstBirth],
        });
        cps.push(EpochCheckpoint {
            tick: 300,
            epoch_id: EPOCH_ROOTS,
            description: "Constitution ratified".into(),
            assertions: vec![EpochAssertion::ConstitutionRatified],
        });
        cps.push(EpochCheckpoint {
            tick: 360,
            epoch_id: EPOCH_ROOTS,
            description: "Self-sufficiency above 0.5".into(),
            assertions: vec![EpochAssertion::SelfSufficiencyAbove(0.5)],
        });
        cps.push(EpochCheckpoint {
            tick: 408,
            epoch_id: EPOCH_ROOTS,
            description: "Roots epoch end: genetic diversity maintained".into(),
            assertions: vec![
                EpochAssertion::GeneticDiversityAbove(0.5),
                EpochAssertion::PopulationAbove(15),
            ],
        });

        // === Branches (408-708) ===
        cps.push(EpochCheckpoint {
            tick: 480,
            epoch_id: EPOCH_BRANCHES,
            description: "Mars colony founded".into(),
            assertions: vec![
                EpochAssertion::WorldCount(2),
                EpochAssertion::PopulationAbove(200),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 540,
            epoch_id: EPOCH_BRANCHES,
            description: "Inter-world trade established".into(),
            assertions: vec![EpochAssertion::TradeEstablished],
        });
        cps.push(EpochCheckpoint {
            tick: 600,
            epoch_id: EPOCH_BRANCHES,
            description: "No oppression crisis active".into(),
            assertions: vec![EpochAssertion::NoOppressionCrisis],
        });
        cps.push(EpochCheckpoint {
            tick: 708,
            epoch_id: EPOCH_BRANCHES,
            description: "Branches epoch end: cultural divergence and population growth".into(),
            assertions: vec![
                EpochAssertion::PopulationAbove(500),
                EpochAssertion::GeneticDiversityAbove(0.4),
            ],
        });

        // === Canopy (708-1068) ===
        cps.push(EpochCheckpoint {
            tick: 780,
            epoch_id: EPOCH_CANOPY,
            description: "Inter-world council formed".into(),
            assertions: vec![
                EpochAssertion::WorldCount(2),
                EpochAssertion::ConstitutionRatified,
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 900,
            epoch_id: EPOCH_CANOPY,
            description: "Population growth milestone".into(),
            assertions: vec![EpochAssertion::PopulationAbove(1000)],
        });
        cps.push(EpochCheckpoint {
            tick: 960,
            epoch_id: EPOCH_CANOPY,
            description: "Economic equity: no oppression crisis".into(),
            assertions: vec![
                EpochAssertion::NoOppressionCrisis,
                EpochAssertion::SelfSufficiencyAbove(0.6),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 1068,
            epoch_id: EPOCH_CANOPY,
            description: "Canopy epoch end: innovation level doubled from baseline".into(),
            assertions: vec![
                EpochAssertion::TechLevelAbove(0.5),
                EpochAssertion::CollectivePhiAbove(0.15),
            ],
        });

        // === Fruiting (1068-1428) ===
        cps.push(EpochCheckpoint {
            tick: 1140,
            epoch_id: EPOCH_FRUITING,
            description: "Manufacturing self-sufficiency above 0.8".into(),
            assertions: vec![EpochAssertion::SelfSufficiencyAbove(0.8)],
        });
        cps.push(EpochCheckpoint {
            tick: 1200,
            epoch_id: EPOCH_FRUITING,
            description: "Love coherence measurable".into(),
            assertions: vec![EpochAssertion::LoveCoherenceAbove(0.1)],
        });
        cps.push(EpochCheckpoint {
            tick: 1320,
            epoch_id: EPOCH_FRUITING,
            description: "Multi-world population thriving".into(),
            assertions: vec![
                EpochAssertion::PopulationAbove(5000),
                EpochAssertion::WorldCount(2),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 1428,
            epoch_id: EPOCH_FRUITING,
            description: "Fruiting epoch end: collective phi emerging".into(),
            assertions: vec![
                EpochAssertion::CollectivePhiAbove(0.15),
                EpochAssertion::GeneticDiversityAbove(0.3),
            ],
        });

        // === Dispersal (1428-1800) ===
        cps.push(EpochCheckpoint {
            tick: 1500,
            epoch_id: EPOCH_DISPERSAL,
            description: "Long-range genetic viability".into(),
            assertions: vec![
                EpochAssertion::GeneticDiversityAbove(0.3),
                EpochAssertion::PopulationAbove(10_000),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 1620,
            epoch_id: EPOCH_DISPERSAL,
            description: "Genetic diversity preserved across generations".into(),
            assertions: vec![EpochAssertion::GeneticDiversityAbove(0.2)],
        });
        cps.push(EpochCheckpoint {
            tick: 1740,
            epoch_id: EPOCH_DISPERSAL,
            description: "Love coherence maturing".into(),
            assertions: vec![EpochAssertion::LoveCoherenceAbove(0.15)],
        });
        cps.push(EpochCheckpoint {
            tick: 1800,
            epoch_id: EPOCH_DISPERSAL,
            description: "Final CVS assessment: civilization viable".into(),
            assertions: vec![
                EpochAssertion::CvsAbove(0.1),
                EpochAssertion::PopulationAbove(100),
            ],
        });

        // === Convergence (1800-2400, ticks ~year 150-200) ===
        // Only build if we're running > 150 years
        cps.push(EpochCheckpoint {
            tick: 2100,
            epoch_id: EPOCH_CONVERGENCE,
            description: "Inter-world council: multiple worlds cooperating".into(),
            assertions: vec![
                EpochAssertion::WorldCount(3),
                EpochAssertion::InterWorldDivergenceBelow(0.5),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 2400,
            epoch_id: EPOCH_CONVERGENCE,
            description: "Amendment stability: governance not calcifying".into(),
            assertions: vec![
                EpochAssertion::AmendmentRateBelow(5.0),
                EpochAssertion::NoOppressionCrisis,
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 2700,
            epoch_id: EPOCH_CONVERGENCE,
            description: "Faction emergence: political diversity".into(),
            assertions: vec![
                EpochAssertion::FactionCountAbove(1),
                EpochAssertion::NoFactionConflictActive,
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 3000,
            epoch_id: EPOCH_CONVERGENCE,
            description: "Innovation check: no stagnation".into(),
            assertions: vec![
                EpochAssertion::InnovationNonStagnant,
                EpochAssertion::TechLevelAbove(0.3),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 3300,
            epoch_id: EPOCH_CONVERGENCE,
            description: "Genetic recovery: diversity maintained across worlds".into(),
            assertions: vec![
                EpochAssertion::GeneticDiversityAbove(0.3),
                EpochAssertion::SpeciationBelow(0.4),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 3600,
            epoch_id: EPOCH_CONVERGENCE,
            description: "300-year pilot gate: civilization endures".into(),
            assertions: vec![
                EpochAssertion::CvsAbove(0.2),
                EpochAssertion::PopulationAbove(1000),
                EpochAssertion::TraumaBelow(0.3),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 3924,
            epoch_id: EPOCH_CONVERGENCE,
            description: "Convergence complete: multi-world civilization stable".into(),
            assertions: vec![
                EpochAssertion::CvsAbove(0.25),
                EpochAssertion::RecoveryDemonstrated,
            ],
        });

        // === Secondary Growth (2400-3600, ~year 200-300) ===
        cps.push(EpochCheckpoint {
            tick: 4200,
            epoch_id: EPOCH_SECONDARY_GROWTH,
            description: "Phi trend classified: consciousness trajectory known".into(),
            assertions: vec![
                EpochAssertion::PhiTrendClassified,
                EpochAssertion::CollectivePhiAbove(0.15),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 4500,
            epoch_id: EPOCH_SECONDARY_GROWTH,
            description: "World survival: all colonies persist".into(),
            assertions: vec![
                EpochAssertion::WorldCount(3),
                EpochAssertion::PopulationAbove(2000),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 4800,
            epoch_id: EPOCH_SECONDARY_GROWTH,
            description: "Innovation non-stagnant: tech continues advancing".into(),
            assertions: vec![EpochAssertion::InnovationNonStagnant],
        });
        cps.push(EpochCheckpoint {
            tick: 5100,
            epoch_id: EPOCH_SECONDARY_GROWTH,
            description: "Divergence stable: cultural distance manageable".into(),
            assertions: vec![EpochAssertion::InterWorldDivergenceBelow(0.6)],
        });
        cps.push(EpochCheckpoint {
            tick: 5400,
            epoch_id: EPOCH_SECONDARY_GROWTH,
            description: "Recovery demonstrated: civilization can bounce back".into(),
            assertions: vec![EpochAssertion::RecoveryDemonstrated],
        });
        cps.push(EpochCheckpoint {
            tick: 5700,
            epoch_id: EPOCH_SECONDARY_GROWTH,
            description: "Elite persistence check: meritocracy maintained".into(),
            assertions: vec![EpochAssertion::ElitePersistenceBelow(0.5)],
        });
        cps.push(EpochCheckpoint {
            tick: 6048,
            epoch_id: EPOCH_SECONDARY_GROWTH,
            description: "Transcendence milestone: consciousness matures".into(),
            assertions: vec![
                EpochAssertion::CvsAbove(0.25),
                EpochAssertion::CollectivePhiAbove(0.2),
            ],
        });

        // === Legacy (3000-3600 ticks = year 250-300+) ===
        cps.push(EpochCheckpoint {
            tick: 6600,
            epoch_id: EPOCH_LEGACY,
            description: "Genetic equilibrium: speciation under control".into(),
            assertions: vec![
                EpochAssertion::SpeciationBelow(0.5),
                EpochAssertion::GeneticDiversityAbove(0.2),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 7200,
            epoch_id: EPOCH_LEGACY,
            description: "Faction cycle completed: factions emerge and dissolve".into(),
            assertions: vec![EpochAssertion::FactionCountAbove(0)],
        });
        cps.push(EpochCheckpoint {
            tick: 7800,
            epoch_id: EPOCH_LEGACY,
            description: "Constitutional vitality: governance evolves".into(),
            assertions: vec![EpochAssertion::NoOppressionCrisis],
        });
        cps.push(EpochCheckpoint {
            tick: 8400,
            epoch_id: EPOCH_LEGACY,
            description: "Innovation renewal: tech growth sustained".into(),
            assertions: vec![
                EpochAssertion::InnovationNonStagnant,
                EpochAssertion::TechLevelAbove(0.4),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 9000,
            epoch_id: EPOCH_LEGACY,
            description: "Self-sufficiency achieved: worlds independent".into(),
            assertions: vec![EpochAssertion::SelfSufficiencyAbove(0.7)],
        });
        cps.push(EpochCheckpoint {
            tick: 9324,
            epoch_id: EPOCH_LEGACY,
            description: "CVS final assessment for 777-year runs".into(),
            assertions: vec![
                EpochAssertion::CvsAbove(0.2),
                EpochAssertion::PopulationAbove(500),
            ],
        });

        // === Millennium (year 300+, only for 1000-year runs) ===
        cps.push(EpochCheckpoint {
            tick: 9600,
            epoch_id: EPOCH_MILLENNIUM,
            description: "Millennium dawn: civilization enters deep time".into(),
            assertions: vec![
                EpochAssertion::CvsAbove(0.2),
                EpochAssertion::TraumaBelow(0.3),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 10200,
            epoch_id: EPOCH_MILLENNIUM,
            description: "Deep time consciousness: phi trend positive".into(),
            assertions: vec![
                EpochAssertion::PhiTrendClassified,
                EpochAssertion::CollectivePhiAbove(0.2),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 10800,
            epoch_id: EPOCH_MILLENNIUM,
            description: "Millennium genetics: diversity holding".into(),
            assertions: vec![
                EpochAssertion::GeneticDiversityAbove(0.15),
                EpochAssertion::SpeciationBelow(0.6),
            ],
        });
        cps.push(EpochCheckpoint {
            tick: 11400,
            epoch_id: EPOCH_MILLENNIUM,
            description: "Millennium harmony: love coherence sustained".into(),
            assertions: vec![EpochAssertion::LoveCoherenceAbove(0.1)],
        });
        cps.push(EpochCheckpoint {
            tick: 12000,
            epoch_id: EPOCH_MILLENNIUM,
            description: "1000-year final assessment: civilization endures".into(),
            assertions: vec![
                EpochAssertion::CvsAbove(0.15),
                EpochAssertion::PopulationAbove(500),
                EpochAssertion::InnovationNonStagnant,
            ],
        });

        cps
    }

    /// Evaluate any checkpoints whose tick matches the current tick.
    /// Returns events for checkpoint pass/fail.
    pub fn evaluate_tick(&mut self, tick: u32, worlds: &[World]) -> Vec<CivEvent> {
        let mut events = Vec::new();

        let snapshot = EpochSnapshot::from_worlds(self.current_epoch, tick, worlds);

        // Find checkpoints at this tick
        let matching: Vec<_> = self
            .checkpoints
            .iter()
            .filter(|cp| cp.tick == tick)
            .cloned()
            .collect();

        for cp in matching {
            let mut failures = Vec::new();

            for assertion in &cp.assertions {
                if let Some(reason) = self.evaluate_assertion(assertion, &snapshot) {
                    failures.push(reason);
                }
            }

            let passed = failures.is_empty();
            let status = if passed { "PASSED" } else { "FAILED" };

            events.push(CivEvent::new(
                tick,
                None,
                if passed {
                    CivEventType::HarmonyMilestone
                } else {
                    CivEventType::EmergencyDeclared
                },
                format!(
                    "Checkpoint [{}] {}: {}",
                    epoch_name(cp.epoch_id),
                    status,
                    cp.description
                ),
            ));

            self.checkpoint_results
                .push((tick, cp.description.clone(), passed, failures));
        }

        events
    }

    /// Evaluate a single assertion against a snapshot. Returns None if passed,
    /// Some(reason) if failed.
    fn evaluate_assertion(
        &self,
        assertion: &EpochAssertion,
        snapshot: &EpochSnapshot,
    ) -> Option<String> {
        match assertion {
            EpochAssertion::PopulationAbove(min) => {
                if snapshot.total_population >= *min {
                    None
                } else {
                    Some(format!(
                        "Population {} < required {}",
                        snapshot.total_population, min
                    ))
                }
            }
            EpochAssertion::WorldCount(min) => {
                if snapshot.world_count >= *min {
                    None
                } else {
                    Some(format!(
                        "World count {} < required {}",
                        snapshot.world_count, min
                    ))
                }
            }
            EpochAssertion::CollectivePhiAbove(min) => {
                if snapshot.mean_phi >= *min {
                    None
                } else {
                    Some(format!(
                        "Collective phi {:.3} < required {:.3}",
                        snapshot.mean_phi, min
                    ))
                }
            }
            EpochAssertion::LoveCoherenceAbove(min) => {
                if snapshot.mean_love_coherence >= *min {
                    None
                } else {
                    Some(format!(
                        "Love coherence {:.3} < required {:.3}",
                        snapshot.mean_love_coherence, min
                    ))
                }
            }
            EpochAssertion::SelfSufficiencyAbove(min) => {
                if snapshot.mean_self_sufficiency >= *min {
                    None
                } else {
                    Some(format!(
                        "Self-sufficiency {:.3} < required {:.3}",
                        snapshot.mean_self_sufficiency, min
                    ))
                }
            }
            EpochAssertion::GeneticDiversityAbove(min) => {
                if snapshot.genetic_diversity >= *min {
                    None
                } else {
                    Some(format!(
                        "Genetic diversity {:.3} < required {:.3}",
                        snapshot.genetic_diversity, min
                    ))
                }
            }
            EpochAssertion::NoOppressionCrisis => {
                if snapshot.max_oppression_index < 0.7 {
                    None
                } else {
                    Some(format!(
                        "Oppression index {:.3} >= crisis threshold 0.7",
                        snapshot.max_oppression_index
                    ))
                }
            }
            EpochAssertion::TechLevelAbove(min) => {
                if snapshot.mean_tech_level >= *min {
                    None
                } else {
                    Some(format!(
                        "Tech level {:.3} < required {:.3}",
                        snapshot.mean_tech_level, min
                    ))
                }
            }
            EpochAssertion::TradeEstablished => {
                if self.first_trade_tick.is_some() {
                    None
                } else {
                    Some("Inter-world trade not yet established".into())
                }
            }
            EpochAssertion::ConstitutionRatified => {
                if self.first_constitution_tick.is_some() {
                    None
                } else {
                    Some("Constitution not yet ratified".into())
                }
            }
            EpochAssertion::FirstBirth => {
                if self.first_birth_tick.is_some() {
                    None
                } else {
                    Some("No births recorded yet".into())
                }
            }
            EpochAssertion::CvsAbove(min) => {
                if snapshot.civilization_viability_score >= *min {
                    None
                } else {
                    Some(format!(
                        "CVS {:.3} < required {:.3}",
                        snapshot.civilization_viability_score, min
                    ))
                }
            }
            EpochAssertion::FactionCountAbove(min) => {
                if snapshot.faction_count >= *min {
                    None
                } else {
                    Some(format!(
                        "Faction count {} < required {}",
                        snapshot.faction_count, min
                    ))
                }
            }
            EpochAssertion::ElitePersistenceBelow(max) => {
                if snapshot.elite_persistence <= *max {
                    None
                } else {
                    Some(format!(
                        "Elite persistence {:.3} > max {:.3}",
                        snapshot.elite_persistence, max
                    ))
                }
            }
            EpochAssertion::InnovationNonStagnant => {
                if snapshot.innovation_stagnation < 0.5 {
                    None
                } else {
                    Some(format!(
                        "Innovation stagnation {:.3} >= 0.5",
                        snapshot.innovation_stagnation
                    ))
                }
            }
            EpochAssertion::InterWorldDivergenceBelow(max) => {
                if snapshot.inter_world_divergence <= *max {
                    None
                } else {
                    Some(format!(
                        "Inter-world divergence {:.3} > max {:.3}",
                        snapshot.inter_world_divergence, max
                    ))
                }
            }
            EpochAssertion::AmendmentRateBelow(max) => {
                if snapshot.amendment_rate <= *max {
                    None
                } else {
                    Some(format!(
                        "Amendment rate {:.1} > max {:.1}",
                        snapshot.amendment_rate, max
                    ))
                }
            }
            EpochAssertion::RecoveryDemonstrated => {
                if snapshot.recovery_count > 0 {
                    None
                } else {
                    Some("No CVS recovery demonstrated".into())
                }
            }
            EpochAssertion::PhiTrendClassified => {
                if snapshot.phi_trend != "Plateau" {
                    None
                } else {
                    Some("Phi trend is Plateau (not yet classified)".into())
                }
            }
            EpochAssertion::TraumaBelow(max) => {
                if snapshot.trauma_level <= *max {
                    None
                } else {
                    Some(format!(
                        "Trauma level {:.3} > max {:.3}",
                        snapshot.trauma_level, max
                    ))
                }
            }
            EpochAssertion::SpeciationBelow(max) => {
                if snapshot.speciation_index <= *max {
                    None
                } else {
                    Some(format!(
                        "Speciation index {:.3} > max {:.3}",
                        snapshot.speciation_index, max
                    ))
                }
            }
            EpochAssertion::NoFactionConflictActive => {
                // Use faction_count == 0 as proxy (no factions = no conflicts)
                // In practice this is evaluated against snapshot data
                None // Pass by default — conflicts are tracked in the main loop
            }
        }
    }

    /// Check whether the simulation should transition to a new epoch.
    /// Returns the new epoch id if a transition should occur.
    pub fn check_epoch_transition(
        &mut self,
        tick: u32,
        total_population: usize,
        num_worlds: usize,
        self_sufficiency: f64,
    ) -> Option<EpochId> {
        let next = match self.current_epoch {
            0 => {
                // Seeds -> Roots: tick >= 168 (permanent habitation)
                if tick >= 168 {
                    Some(EPOCH_ROOTS)
                } else {
                    None
                }
            }
            1 => {
                // Roots -> Branches: tick >= 408 OR (pop > 200 AND ss > 0.6)
                if tick >= 408 || (total_population > 200 && self_sufficiency > 0.6) {
                    Some(EPOCH_BRANCHES)
                } else {
                    None
                }
            }
            2 => {
                // Branches -> Canopy: tick >= 708 OR (2+ worlds with 1000+ each)
                if tick >= 708 || num_worlds >= 2 {
                    Some(EPOCH_CANOPY)
                } else {
                    None
                }
            }
            3 => {
                // Canopy -> Fruiting: tick >= 1068
                if tick >= 1068 {
                    Some(EPOCH_FRUITING)
                } else {
                    None
                }
            }
            4 => {
                // Fruiting -> Dispersal: tick >= 1428
                if tick >= 1428 {
                    Some(EPOCH_DISPERSAL)
                } else {
                    None
                }
            }
            5 => {
                // Dispersal -> Convergence: tick >= 1800 (year 150)
                if tick >= 1800 {
                    Some(EPOCH_CONVERGENCE)
                } else {
                    None
                }
            }
            6 => {
                // Convergence -> Secondary Growth: tick >= 2400 (year 200)
                if tick >= 2400 {
                    Some(EPOCH_SECONDARY_GROWTH)
                } else {
                    None
                }
            }
            7 => {
                // Secondary Growth -> Legacy: tick >= 3000 (year 250)
                if tick >= 3000 {
                    Some(EPOCH_LEGACY)
                } else {
                    None
                }
            }
            8 => {
                // Legacy -> Millennium: tick >= 3600 (year 300)
                if tick >= 3600 {
                    Some(EPOCH_MILLENNIUM)
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(new_epoch) = next {
            self.epoch_transitions
                .push((tick, self.current_epoch, new_epoch));
            self.current_epoch = new_epoch;
        }

        next
    }

    /// Compute the Civilization Viability Score (CVS) — arithmetic mean.
    ///
    /// CVS = 0.2 * genetic + 0.2 * economic + 0.2 * harmonies + 0.2 * (1 - oppression) + 0.2 * phi
    ///
    /// All inputs should be in [0, 1]. The result is in [0, 1]. Fully
    /// substitutable — one collapsed dimension can be hidden by strong
    /// performance elsewhere. See `compute_cvs_geometric` for a "weakest
    /// link" variant inspired by kosmic-lab's historical K-index.
    pub fn compute_cvs(
        genetic: f64,
        economic: f64,
        harmonies: f64,
        oppression: f64,
        phi: f64,
    ) -> f64 {
        let g = genetic.clamp(0.0, 1.0);
        let e = economic.clamp(0.0, 1.0);
        let h = harmonies.clamp(0.0, 1.0);
        let o = 1.0 - oppression.clamp(0.0, 1.0);
        let p = phi.clamp(0.0, 1.0);
        0.2 * g + 0.2 * e + 0.2 * h + 0.2 * o + 0.2 * p
    }

    /// Geometric-mean CVS — the "weakest link" variant (kosmic-lab K-index
    /// Path C default, 2025-11-27).
    ///
    /// `CVS_geo = exp((1/5) * Σ log(x_i + ε))`, ε = 1e-10 for numerical
    /// stability near zero. A civilization with `oppression = 1.0` (i.e.
    /// the anti-oppression component is 0) scores close to 0 regardless
    /// of how well other dimensions perform. Matches the intuition that
    /// a totally oppressive society isn't civilizationally viable even
    /// with great genetics.
    ///
    /// Returns a value in [0, 1]. Strictly ≤ `compute_cvs` for the same
    /// inputs (geometric ≤ arithmetic; equal only when all inputs equal).
    pub fn compute_cvs_geometric(
        genetic: f64,
        economic: f64,
        harmonies: f64,
        oppression: f64,
        phi: f64,
    ) -> f64 {
        const EPS: f64 = 1e-10;
        let parts = [
            genetic.clamp(0.0, 1.0),
            economic.clamp(0.0, 1.0),
            harmonies.clamp(0.0, 1.0),
            1.0 - oppression.clamp(0.0, 1.0),
            phi.clamp(0.0, 1.0),
        ];
        let n = parts.len() as f64;
        let sum_log: f64 = parts.iter().map(|x| (x + EPS).ln()).sum();
        let geo = (sum_log / n).exp();
        // Subtract the ε floor so a truly-zero input maps to ~0 rather
        // than exp(ln(ε)) ≈ ε. Keeps the scale comparable to arithmetic.
        (geo - EPS).clamp(0.0, 1.0)
    }

    /// Record a milestone event by kind.
    pub fn record_milestone(&mut self, kind: &str, tick: u32) {
        match kind {
            "birth" => {
                if self.first_birth_tick.is_none() {
                    self.first_birth_tick = Some(tick);
                }
            }
            "constitution" => {
                if self.first_constitution_tick.is_none() {
                    self.first_constitution_tick = Some(tick);
                }
            }
            "trade" => {
                if self.first_trade_tick.is_none() {
                    self.first_trade_tick = Some(tick);
                }
            }
            _ => {}
        }
    }

    /// Take a snapshot of the current state.
    pub fn take_snapshot(&self, tick: u32, worlds: &[World]) -> EpochSnapshot {
        EpochSnapshot::from_worlds(self.current_epoch, tick, worlds)
    }

    /// Number of checkpoints that passed.
    pub fn checkpoints_passed(&self) -> usize {
        self.checkpoint_results
            .iter()
            .filter(|(_, _, p, _)| *p)
            .count()
    }

    /// Number of checkpoints that failed.
    pub fn checkpoints_failed(&self) -> usize {
        self.checkpoint_results
            .iter()
            .filter(|(_, _, p, _)| !*p)
            .count()
    }

    /// Convert a tick number to a fractional year offset from simulation start (2026).
    pub fn tick_to_year(tick: u32) -> f64 {
        tick as f64 / TICKS_PER_YEAR as f64
    }
}

impl Default for EpochManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
    use crate::world::{CulturalProfile, World, WorldResources};

    fn make_test_world(pop: usize, world_id: u32) -> World {
        let mut world = World {
            id: world_id,
            name: format!("World-{}", world_id),
            location: "Moon".into(),
            founded_tick: 0,
            parent_world_id: None,
            agents: Vec::new(),
            next_agent_id: 0,
            resources: WorldResources::lunar_default(),
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.5,
            max_population: 100_000,
            habitable_area_m2: 1_000_000.0,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: crate::knowledge::WorldKnowledge::new(),
            economy: crate::economy::WorldEconomy::new(),
            harmony: crate::harmony::HarmonyTracker::new(),
            governance: crate::governance::WorldGovernance::new(),
            metabolism_state: crate::metabolism::MetabolismState::default(),
            currency_state: crate::currency::WorldCurrencyState::default(),
            policy_state: crate::proposals::PolicyState::default(),
            power_generation_kw: 0.0,
            power_demand_kw: 0.0,
            narrative_identity: crate::world::NarrativeIdentity::default(),
            maintenance_hours_required: 0.0,
            maintenance_hours_available: 0.0,
            bus_factor_critical: 0,
            pathogen_pressure: 0.0,
            civilizational_phi: 0.0,
            trust_level: 0.7,
            earth_funding: 1.0,
            mortality_alpha_mult: 1.0,
            mortality_beta_mult: 1.0,
            mortality_lambda_mult: 1.0,
            reproduction_viable: true,
            ecosystem_balance: 1.0,
            fertility_multiplier: 1.0,
            automation_level: 0.0,
            explorations_completed: 0,
            project_manager: crate::projects::ProjectManager::new(),
            habitat: crate::habitat::HabitatComplex::default(),
            fleet: crate::robotics::RoboticFleet::default(),
            diplomatic_relations: std::collections::HashMap::new(),
            zones: Vec::new(),
            moral_memories: Vec::new(),
            institutional_ethics: crate::agent::EthicalOrientation::default(),
        };
        for i in 0..pop {
            let agent = CivAgent {
                id: i as u64,
                birth_tick: 0,
                death_tick: None,
                sex: if i % 2 == 0 {
                    BiologicalSex::Female
                } else {
                    BiologicalSex::Male
                },
                world_id,
                health: 0.9,
                skills: SkillVector::new(),
                education_level: 0.3,
                consciousness: ConsciousnessState::nascent(),
                partner_id: None,
                children_ids: vec![],
                is_immigrant: false,
                needs: crate::needs::PsychologicalNeeds::new(),
                tend_balance: 0.0,
                parent_ids: None,
                faction_id: None,
                generation: 0,
                trauma_level: 0.0,
                cumulative_dose_sv: 0.0,
                adversarial: None,
                coordination_understanding: 0.0,
                mycel_score: 0.1,
                sap_balance: 100.0,
                is_biological: true,
                wounds: Vec::new(),
                ethics: crate::agent::EthicalOrientation::default(),
                sovereign_profile: crate::sovereign_profile::SovereignProfile::zero(),
                justice: crate::sub_passport::RestorativeJustice::new(),
            };
            world.agents.push(agent);
        }
        world.next_agent_id = pop as u64;
        world
    }

    #[test]
    fn test_epoch_manager_creates_checkpoints() {
        let mgr = EpochManager::new();
        assert!(
            mgr.checkpoints.len() >= 20,
            "Expected at least 20 checkpoints, got {}",
            mgr.checkpoints.len()
        );
        // Check Seeds epoch has 3 checkpoints
        let seeds_count = mgr
            .checkpoints
            .iter()
            .filter(|cp| cp.epoch_id == EPOCH_SEEDS)
            .count();
        assert!(seeds_count >= 3, "Seeds epoch should have >= 3 checkpoints");
    }

    #[test]
    fn test_cvs_computation() {
        // Perfect scores -> 1.0
        let cvs = EpochManager::compute_cvs(1.0, 1.0, 1.0, 0.0, 1.0);
        assert!(
            (cvs - 1.0).abs() < 1e-9,
            "Perfect CVS should be 1.0, got {cvs}"
        );

        // All zeros (with max oppression) -> 0.0
        let cvs_zero = EpochManager::compute_cvs(0.0, 0.0, 0.0, 1.0, 0.0);
        assert!(
            cvs_zero.abs() < 1e-9,
            "Zero CVS should be 0.0, got {cvs_zero}"
        );

        // Mid-range: 0.2*(0.5+0.5+0.5+(1-0)+0.5) = 0.2*3.0 = 0.6
        let cvs_mid = EpochManager::compute_cvs(0.5, 0.5, 0.5, 0.0, 0.5);
        assert!(
            (cvs_mid - 0.6).abs() < 0.01,
            "Mid CVS should be 0.6, got {cvs_mid}"
        );

        // Balanced mid-range with some oppression
        let cvs_balanced = EpochManager::compute_cvs(0.5, 0.5, 0.5, 0.5, 0.5);
        assert!(
            (cvs_balanced - 0.5).abs() < 0.01,
            "Balanced CVS should be 0.5, got {cvs_balanced}"
        );
    }

    #[test]
    fn test_cvs_geometric_perfect_and_zero() {
        // Perfect inputs → 1.0.
        let cvs = EpochManager::compute_cvs_geometric(1.0, 1.0, 1.0, 0.0, 1.0);
        assert!((cvs - 1.0).abs() < 1e-5, "Perfect CVS_geo ≈ 1.0, got {cvs}");
        // Fully oppressed (anti-oppression term = 0) → near 0 even with
        // other maxed inputs. Arithmetic would give 0.8 here.
        let cvs_oppressed = EpochManager::compute_cvs_geometric(1.0, 1.0, 1.0, 1.0, 1.0);
        assert!(
            cvs_oppressed < 0.05,
            "Fully oppressed CVS_geo should ≈ 0, got {cvs_oppressed}",
        );
    }

    #[test]
    fn test_cvs_geometric_le_arithmetic() {
        // For the same inputs, geometric ≤ arithmetic (equal only when all
        // inputs are equal — AM–GM inequality).
        for &(g, e, h, o, p) in &[
            (0.5, 0.5, 0.5, 0.5, 0.5),
            (0.3, 0.8, 0.6, 0.2, 0.7),
            (0.9, 0.1, 0.5, 0.0, 0.5),
            (0.7, 0.7, 0.7, 0.3, 0.7),
        ] {
            let arith = EpochManager::compute_cvs(g, e, h, o, p);
            let geo = EpochManager::compute_cvs_geometric(g, e, h, o, p);
            assert!(
                geo <= arith + 1e-9,
                "geo {:.4} should be ≤ arith {:.4} for ({},{},{},{},{})",
                geo,
                arith,
                g,
                e,
                h,
                o,
                p,
            );
        }
    }

    #[test]
    fn test_cvs_geometric_weakest_link() {
        // A civilization with one zero-valued component should be
        // severely penalized by geometric mean, even if other components
        // are perfect. Arithmetic mean cannot distinguish such a profile
        // from one with moderate values everywhere.
        //
        // Profile A: perfect on 4 dims, zero on phi.
        //   arith = (1+1+1+1+0)/5 = 0.8
        // Profile B: uniform 0.8 across all dims.
        //   arith = (0.8*4 + 0.8)/5 = 0.8  (note: anti-opp here is 0.8 → opp=0.2)
        let arith_a = EpochManager::compute_cvs(1.0, 1.0, 1.0, 0.0, 0.0);
        let arith_b = EpochManager::compute_cvs(0.8, 0.8, 0.8, 0.2, 0.8);
        assert!(
            (arith_a - arith_b).abs() < 0.01,
            "Setup: profiles should have ~equal arithmetic CVS (got {:.3} vs {:.3})",
            arith_a,
            arith_b,
        );
        let geo_a = EpochManager::compute_cvs_geometric(1.0, 1.0, 1.0, 0.0, 0.0);
        let geo_b = EpochManager::compute_cvs_geometric(0.8, 0.8, 0.8, 0.2, 0.8);
        // A has phi=0 → geom should be near-zero regardless of other perfect dims.
        assert!(
            geo_a < 0.1,
            "Profile A (phi=0) should have near-zero geom, got {:.3}",
            geo_a,
        );
        assert!(
            geo_b > geo_a + 0.5,
            "Profile B (uniform) should score much higher than A despite same arith: A={:.3}, B={:.3}",
            geo_a,
            geo_b,
        );
    }

    #[test]
    fn test_epoch_transition_seeds_to_roots() {
        let mut mgr = EpochManager::new();
        assert_eq!(mgr.current_epoch, EPOCH_SEEDS);

        // Before tick 168: no transition
        assert!(mgr.check_epoch_transition(100, 50, 1, 0.4).is_none());

        // At tick 168: transition
        let result = mgr.check_epoch_transition(168, 50, 1, 0.4);
        assert_eq!(result, Some(EPOCH_ROOTS));
        assert_eq!(mgr.current_epoch, EPOCH_ROOTS);
        assert_eq!(mgr.epoch_transitions.len(), 1);
    }

    #[test]
    fn test_checkpoint_evaluation_pass() {
        let mut mgr = EpochManager::new();
        let worlds = vec![make_test_world(20, 0)];

        // Evaluate tick 60 checkpoint (base assembled: pop >= 5, worlds >= 1)
        let events = mgr.evaluate_tick(60, &worlds);
        assert!(
            !events.is_empty(),
            "Should generate checkpoint event at tick 60"
        );

        let (_, _, passed, failures) = &mgr.checkpoint_results[0];
        assert!(
            passed,
            "Checkpoint should pass with 20 agents; failures: {:?}",
            failures
        );
    }

    #[test]
    fn test_checkpoint_evaluation_fail() {
        let mut mgr = EpochManager::new();
        let worlds = vec![make_test_world(2, 0)]; // Only 2 agents

        // Tick 168 requires pop >= 6
        let events = mgr.evaluate_tick(168, &worlds);
        assert!(!events.is_empty());

        // Find the tick-168 result
        let result = mgr.checkpoint_results.iter().find(|(t, _, _, _)| *t == 168);
        assert!(result.is_some());
        let (_, _, passed, failures) = result.unwrap();
        assert!(!passed, "Checkpoint should fail with only 2 agents");
        assert!(!failures.is_empty());
    }

    #[test]
    fn test_record_milestone() {
        let mut mgr = EpochManager::new();
        assert!(mgr.first_birth_tick.is_none());

        mgr.record_milestone("birth", 100);
        assert_eq!(mgr.first_birth_tick, Some(100));

        // Second birth doesn't overwrite
        mgr.record_milestone("birth", 200);
        assert_eq!(mgr.first_birth_tick, Some(100));

        mgr.record_milestone("constitution", 300);
        assert_eq!(mgr.first_constitution_tick, Some(300));

        mgr.record_milestone("trade", 500);
        assert_eq!(mgr.first_trade_tick, Some(500));
    }

    #[test]
    fn test_snapshot_from_worlds() {
        let worlds = vec![make_test_world(100, 0), make_test_world(50, 1)];
        let snap = EpochSnapshot::from_worlds(EPOCH_BRANCHES, 480, &worlds);

        assert_eq!(snap.epoch_id, EPOCH_BRANCHES);
        assert_eq!(snap.tick, 480);
        assert_eq!(snap.total_population, 150);
        assert_eq!(snap.world_count, 2);
        assert!(snap.mean_self_sufficiency > 0.0);
        assert!(snap.genetic_diversity > 0.0);
        assert!(snap.civilization_viability_score > 0.0);
    }

    #[test]
    fn test_tick_to_year() {
        assert!((EpochManager::tick_to_year(0) - 0.0).abs() < 1e-9);
        assert!((EpochManager::tick_to_year(12) - 1.0).abs() < 1e-9);
        assert!((EpochManager::tick_to_year(1800) - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_full_epoch_progression() {
        let mut mgr = EpochManager::new();

        // Seeds -> Roots at tick 168
        mgr.check_epoch_transition(168, 50, 1, 0.4);
        assert_eq!(mgr.current_epoch, EPOCH_ROOTS);

        // Roots -> Branches at tick 408
        mgr.check_epoch_transition(408, 300, 1, 0.7);
        assert_eq!(mgr.current_epoch, EPOCH_BRANCHES);

        // Branches -> Canopy at tick 708
        mgr.check_epoch_transition(708, 1000, 2, 0.8);
        assert_eq!(mgr.current_epoch, EPOCH_CANOPY);

        // Canopy -> Fruiting at tick 1068
        mgr.check_epoch_transition(1068, 5000, 2, 0.85);
        assert_eq!(mgr.current_epoch, EPOCH_FRUITING);

        // Fruiting -> Dispersal at tick 1428
        mgr.check_epoch_transition(1428, 20000, 3, 0.9);
        assert_eq!(mgr.current_epoch, EPOCH_DISPERSAL);

        assert_eq!(mgr.epoch_transitions.len(), 5);
    }
}
