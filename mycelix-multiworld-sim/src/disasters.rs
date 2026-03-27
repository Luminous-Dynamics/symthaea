// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Probabilistic disaster engine with 40 events across 7 categories.
//!
//! All probabilities are grounded in published scientific data:
//!
//! - **Solar**: NOAA SWPC historical flare rates; Riley (2012) Carrington estimate
//!   0.7%/year; Usoskin et al. (2012) SPE reconstruction from ice cores.
//! - **Impact**: Gruen et al. (1985) micrometeorite flux; Ceplecha et al. (1998)
//!   bolide frequency-size distribution.
//! - **Planetary**: Zurek & Martin (1993) Mars dust storm climatology; Nakamura
//!   et al. (1982) Apollo Passive Seismic Experiment moonquake catalog.
//! - **ECLSS**: ISS ECLSS subsystem MTBF data (NASA TM-2005-214062); Wieland
//!   (1998) "Designing for Human Presence in Space".
//! - **Psychological**: Palinkas & Suedfeld (2008) Antarctic analogues; Basner
//!   et al. (2014) Mars-500 cognitive study; Sandal et al. (2006) confinement.
//! - **Technology**: NASA Technology Roadmaps (2015); Meier (2022) fusion timeline
//!   estimates; NRC "Pathways to Exploration" (2014).
//! - **Civilization**: Tainter (1988) "Collapse of Complex Societies"; Turchin
//!   (2003) "Historical Dynamics"; Turchin & Nefedov (2009) "Secular Cycles".

use crate::config::PolicyConfig;
use crate::events::{CivEvent, CivEventType};
use crate::stochastic::StochasticEngine;
use crate::world::World;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants: monthly probabilities derived from published annualized rates
// ---------------------------------------------------------------------------

/// Solar cycle period in ticks (11 years * 12 months).
const SOLAR_CYCLE_TICKS: u32 = 132;

// Solar & Space Weather — per-tick (monthly) probabilities
/// Impactful M-class flare: ~10/year but only ~10% cause significant disruption.
const P_M_CLASS_FLARE: f64 = 0.05;
/// Damaging X-class flare: ~1/year, ~12% cause ground-level effects.
const P_X_CLASS_FLARE: f64 = 0.01;
/// Carrington-class event: 0.7%/year (Riley 2012) → ~0.00058/month.
const P_CARRINGTON: f64 = 0.000_58;
/// Major SPE (≥Aug 1972 level): 19 per 450 years (Usoskin 2012) → ~0.0035/month.
const P_MAJOR_SPE: f64 = 0.003_5;

// Impact Events
/// Small meteorite impact on colony footprint: ~0.001%/year → ~8.3e-7/month.
const P_SMALL_METEORITE: f64 = 8.3e-7;
/// Large meteorite: ~0.0001%/year → ~8.3e-8/month.
const P_LARGE_METEORITE: f64 = 8.3e-8;
/// Micrometeorite cumulative degradation per tick (fraction of max damage).
const MICROMETEORITE_DEGRADATION_PER_TICK: f64 = 0.000_167; // 0.2%/year / 12

// Planetary Environment
/// Mars global dust storm: ~1 per 3 Mars years (66 months). Zurek & Martin 1993.
const P_MARS_GLOBAL_DUST: f64 = 0.015;
/// Mars regional dust storm (significant): ~8 per Mars year, ~50% impactful.
const P_MARS_REGIONAL_DUST: f64 = 0.05;
/// Damaging shallow moonquake (M5+): ~5/year, ~5% structurally damaging.
const P_DAMAGING_MOONQUAKE: f64 = 0.02;

// ECLSS / Infrastructure — p = 1 - e^(-1/MTBF) ≈ 1/MTBF for small values
/// O2 generator: MTBF 96 months.
const P_O2_FAILURE: f64 = 0.0104;
/// Water recycler: MTBF 60 months.
const P_WATER_FAILURE: f64 = 0.0167;
/// CO2 scrubber: MTBF 72 months.
const P_CO2_FAILURE: f64 = 0.0139;
/// Thermal control: MTBF 120 months.
const P_THERMAL_FAILURE: f64 = 0.0083;
/// Power distribution: MTBF 84 months.
const P_POWER_FAILURE: f64 = 0.0119;
/// Seal degradation: cumulative 0.1%/year.
const SEAL_DEGRADATION_PER_TICK: f64 = 0.001 / 12.0;
/// Hydroponic system: MTBF 48 months.
const P_HYDROPONIC_FAILURE: f64 = 0.0208;

// Psychological Events — per-tick probabilities for confined crews
/// Winter-over syndrome: ~40% prevalence/year in confined crews (Palinkas 2008).
const P_WINTER_OVER: f64 = 0.04;
/// Interpersonal conflict (significant): ~2/year in small crews (Sandal 2006).
const P_INTERPERSONAL_CONFLICT: f64 = 0.017;
/// Cognitive impairment episode: documented in Mars-500 (Basner 2014).
const P_COGNITIVE_IMPAIRMENT: f64 = 0.008;
/// Social cohesion collapse: rare, requires low-cohesion preconditions.
const P_COHESION_COLLAPSE_BASE: f64 = 0.002;
/// Psychotic break: <0.1% with screening (Kanas 2015).
const P_PSYCHOTIC_BREAK: f64 = 0.000_8;
/// Authority challenge: extremely rare.
const P_AUTHORITY_CHALLENGE: f64 = 0.001;

// Tainter/Turchin thresholds
/// Infrastructure level above which diminishing returns on complexity kick in.
const TAINTER_COMPLEXITY_THRESHOLD: f64 = 0.8;
/// Consciousness Gini above which elite overproduction risk rises.
const TURCHIN_GINI_THRESHOLD: f64 = 0.4;
/// Guardian fraction above which elite overproduction risk rises.
const TURCHIN_GUARDIAN_FRACTION: f64 = 0.15;
/// Constitutional calcification above which institutional sclerosis triggers.
const SCLEROSIS_CALCIFICATION_THRESHOLD: f64 = 0.8;
/// Mean allostatic load above which social cohesion crisis may trigger.
const COHESION_CRISIS_LOAD: f64 = 0.6;
/// Number of ticks of sustained high load before crisis triggers.
const COHESION_CRISIS_TICKS: u32 = 24;
/// Number of infrastructure failures in a window that triggers cascade.
const CASCADE_FAILURE_THRESHOLD: u32 = 3;
/// Window (ticks) for cascade detection.
const CASCADE_WINDOW_TICKS: u32 = 6;
/// Complexity cost history window (10 years).
const COMPLEXITY_HISTORY_WINDOW: usize = 120;

// Tech tree timing (in ticks)
const NTP_EARLIEST: u32 = 24;
const NTP_LATEST: u32 = 60;
const FISSION_EARLIEST: u32 = 36;
const FISSION_LATEST: u32 = 120;
const FUSION_DEMO_EARLIEST: u32 = 120;
const FUSION_DEMO_LATEST: u32 = 240;
const MANUFACTURING_EARLIEST: u32 = 360;
const MANUFACTURING_LATEST: u32 = 720;
const LCF_EARLIEST: u32 = 480;
const LCF_LATEST: u32 = 1200;
const FUSION_GRID_EARLIEST: u32 = 300;
const FUSION_GRID_LATEST: u32 = 600;

// ---------------------------------------------------------------------------
// Event kinds
// ---------------------------------------------------------------------------

/// Solar and space weather event classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SolarEventKind {
    /// ~100/year, mild. 2x SEU rate, 12h duration. NOAA SWPC.
    CClassFlare,
    /// ~10/year impactful. 10x SEU, 24h. ~20% packet loss.
    MClassFlare,
    /// ~1/year severe. 100x SEU, 48h. Potential lethal dose without shelter.
    XClassFlare,
    /// 0.7%/year (Riley 2012). Electronics destruction, weeks of recovery.
    CarringtonEvent,
    /// 19 per 450 years at Aug-1972 level (Usoskin 2012). Acute radiation.
    SolarProtonEvent,
    /// 11-year cycle onset. 15% higher GCR for ~60 ticks. Chronic.
    SolarMinimumOnset,
}

/// Meteorite / impact event classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImpactEventKind {
    /// Cumulative: 2/m^2/year (Gruen 1985). Infrastructure degradation.
    MicrometeoriteBarrage,
    /// ~0.001%/year for colony footprint. Hull breach risk.
    SmallMeteorite,
    /// ~0.0001%/year. Catastrophic structural damage.
    LargeMeteorite,
}

/// Planetary environment event classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlanetaryEventKind {
    /// 1 per ~66 months (3 Mars years). Solar -90%, 1-3 months. Zurek 1993.
    MarsGlobalDustStorm,
    /// ~8 per 26 months (Mars year). Solar -50%, 2-4 weeks.
    MarsRegionalDustStorm,
    /// ~5/year, M5+ possible. Structural stress. Nakamura 1982.
    ShallowMoonquake,
    /// 2x daily, negligible individually. Cumulative fatigue.
    ThermalMoonquake,
    /// Charged dust during terminator crossing. Equipment fouling.
    LunarDustEvent,
}

/// ECLSS and infrastructure failure classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InfrastructureFailureKind {
    /// MTBF ~96 months. Cell stack failure. NASA TM-2005-214062.
    O2GeneratorDegradation,
    /// MTBF ~60 months. Membrane fouling.
    WaterRecyclerFailure,
    /// MTBF ~72 months. Filter clogging.
    CO2ScrubberDegradation,
    /// MTBF ~120 months. Radiator/pump failure.
    ThermalControlFailure,
    /// MTBF ~84 months. Circuit/inverter failure.
    PowerDistributionFault,
    /// Cumulative: 0.1%/year habitat seal aging.
    SealIntegrityDegradation,
    /// MTBF ~48 months. Nutrient/pH imbalance.
    HydroponicSystemFailure,
}

/// Psychological / social event classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PsychologicalEventKind {
    /// 40% of confined crew. Seasonal depression. Palinkas 2008.
    WinterOverSyndrome,
    /// High probability in small groups. Productivity loss. Sandal 2006.
    InterpersonalConflict,
    /// Documented in Mars-500. Decision quality degradation. Basner 2014.
    CognitiveImpairment,
    /// Low-cohesion groups. Triggers faction emergence.
    SocialCohesionCollapse,
    /// <0.1% with screening. Individual crisis. Kanas 2015.
    PsychoticBreak,
    /// Extremely rare. Leadership crisis.
    AuthorityChallenge,
}

/// Technology event classification (positive and negative).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TechEventKind {
    /// Year 2-5. Enables faster transit. NASA TRL 4-5.
    NtpDemonstration,
    /// Year 3-10. Eliminates lunar night crisis. Kilopower heritage.
    FissionSurfacePower,
    /// Year 10-20. Medium confidence. Meier 2022.
    FusionDemo,
    /// Year 25-50. Low-medium confidence.
    FusionGridScale,
    /// Epoch 3-4. 0.5%/tick. Room-temperature fusion.
    LcfBreakthrough,
    /// Year 10-20. Enables ISRU scaling.
    MegawattReactor,
    /// When researchers < critical mass. Knowledge loss.
    TechRegression,
    /// Year 30-60. Self-replicating manufacturing.
    ManufacturingBreakthrough,
}

/// Civilization-scale dynamics (Tainter 1988, Turchin 2003).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CivilizationEventKind {
    /// When infrastructure_level > 0.8 and tech stagnant. Tainter 1988.
    DiminishingReturnsOnComplexity,
    /// When consciousness_gini > 0.4 and guardian_frac > 0.15. Turchin 2003.
    EliteOverproduction,
    /// When any critical resource < 10% for 12+ ticks.
    ResourceDepletionCrisis,
    /// When constitutional_calcification > 0.8.
    InstitutionalSclerosis,
    /// When 3+ infrastructure failures within 6 ticks.
    SystemicCascadeFailure,
    /// When mean allostatic_load > 0.6 for 24+ ticks.
    SocialCohesionCrisis,
}

/// Unified disaster kind covering all 7 categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DisasterKind {
    Solar(SolarEventKind),
    Impact(ImpactEventKind),
    Planetary(PlanetaryEventKind),
    Infrastructure(InfrastructureFailureKind),
    Psychological(PsychologicalEventKind),
    Technology(TechEventKind),
    Civilization(CivilizationEventKind),
}

// ---------------------------------------------------------------------------
// Solar cycle
// ---------------------------------------------------------------------------

/// Phase within the 11-year solar cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SolarCyclePhase {
    Minimum,
    Rising,
    Maximum,
    Declining,
}

impl SolarCyclePhase {
    /// Flare rate multiplier relative to cycle-average.
    pub fn flare_multiplier(&self) -> f64 {
        match self {
            SolarCyclePhase::Minimum => 0.2,
            SolarCyclePhase::Rising => 0.8,
            SolarCyclePhase::Maximum => 2.5,
            SolarCyclePhase::Declining => 1.0,
        }
    }

    /// GCR (galactic cosmic ray) multiplier — inversely correlated with activity.
    pub fn gcr_multiplier(&self) -> f64 {
        match self {
            SolarCyclePhase::Minimum => 1.15,
            SolarCyclePhase::Rising => 1.0,
            SolarCyclePhase::Maximum => 0.8,
            SolarCyclePhase::Declining => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Effects
// ---------------------------------------------------------------------------

/// Aggregated effects of a disaster on a world.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DisasterEffects {
    /// Immediate population loss as a fraction of current population.
    pub population_loss_fraction: f64,
    /// Reduction in infrastructure_level.
    pub infrastructure_damage: f64,
    /// Multiplicative penalty on all resource production (0.0 = no penalty).
    pub resource_production_penalty: f64,
    /// Reduction in solar power output (0.0-1.0).
    pub solar_power_penalty: f64,
    /// Reduction in collective Phi.
    pub consciousness_shock: f64,
    /// Stress increase on all agents.
    pub allostatic_load_increase: f64,
    /// Fraction of electronics/tech equipment damaged.
    pub electronics_damage: f64,
    /// Morale impact (negative = harmful).
    pub morale_impact: f64,
}

impl DisasterEffects {
    /// Combine two effect sets (additive, clamped).
    pub fn merge(&mut self, other: &DisasterEffects) {
        self.population_loss_fraction =
            (self.population_loss_fraction + other.population_loss_fraction).min(1.0);
        self.infrastructure_damage =
            (self.infrastructure_damage + other.infrastructure_damage).min(1.0);
        self.resource_production_penalty =
            (self.resource_production_penalty + other.resource_production_penalty).min(1.0);
        self.solar_power_penalty =
            (self.solar_power_penalty + other.solar_power_penalty).min(1.0);
        self.consciousness_shock =
            (self.consciousness_shock + other.consciousness_shock).min(1.0);
        self.allostatic_load_increase =
            (self.allostatic_load_increase + other.allostatic_load_increase).min(1.0);
        self.electronics_damage =
            (self.electronics_damage + other.electronics_damage).min(1.0);
        self.morale_impact = (self.morale_impact + other.morale_impact).clamp(-1.0, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Active disaster tracking
// ---------------------------------------------------------------------------

/// An ongoing disaster with remaining duration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveDisaster {
    pub kind: DisasterKind,
    pub severity: f64,
    pub remaining_ticks: u32,
    /// Target world (None = affects all worlds).
    pub world_id: Option<u32>,
    pub effects: DisasterEffects,
}

// ---------------------------------------------------------------------------
// Tech tree
// ---------------------------------------------------------------------------

/// A technology milestone that can be achieved during the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechMilestone {
    pub name: String,
    /// Can't happen before this tick.
    pub earliest_tick: u32,
    /// Irrelevant after this tick (overtaken by events).
    pub latest_tick: u32,
    /// Per-tick probability once eligible.
    pub base_probability: f64,
    /// (sector_index, min_level) prerequisites.
    pub prerequisites: Vec<(usize, f64)>,
    /// Named prerequisite milestones that must be achieved first.
    pub prerequisite_milestones: Vec<String>,
    pub effects: TechEffects,
    pub achieved: bool,
    pub achieved_tick: Option<u32>,
}

/// Effects of achieving a tech milestone.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TechEffects {
    /// (sector_index, boost) — additive boost to technology level.
    pub tech_level_boost: Vec<(usize, f64)>,
    /// Multiplier on power output (1.0 = no change).
    pub power_multiplier: f64,
    /// Enables new world founding (propulsion unlock).
    pub propulsion_unlock: bool,
    /// Multiplier on resource production efficiency.
    pub resource_efficiency: f64,
}

/// Complete tech tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechTree {
    pub milestones: Vec<TechMilestone>,
}

impl TechTree {
    /// Default tech tree based on NASA/DOE technology roadmaps.
    pub fn default_tree() -> Self {
        Self {
            milestones: vec![
                TechMilestone {
                    name: "NTP Demonstration".into(),
                    earliest_tick: NTP_EARLIEST,
                    latest_tick: NTP_LATEST,
                    base_probability: 0.02,
                    prerequisites: vec![(1, 1.2)], // engineering > 1.2
                    prerequisite_milestones: vec![],
                    effects: TechEffects {
                        tech_level_boost: vec![(1, 0.3)],
                        power_multiplier: 1.0,
                        propulsion_unlock: true,
                        resource_efficiency: 1.0,
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                TechMilestone {
                    name: "Fission Surface Power".into(),
                    earliest_tick: FISSION_EARLIEST,
                    latest_tick: FISSION_LATEST,
                    base_probability: 0.015,
                    prerequisites: vec![(1, 1.5)], // engineering > 1.5
                    prerequisite_milestones: vec![],
                    effects: TechEffects {
                        tech_level_boost: vec![(1, 0.5), (4, 0.2)],
                        power_multiplier: 2.0,
                        propulsion_unlock: false,
                        resource_efficiency: 1.2,
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                TechMilestone {
                    name: "Fusion Demo".into(),
                    earliest_tick: FUSION_DEMO_EARLIEST,
                    latest_tick: FUSION_DEMO_LATEST,
                    base_probability: 0.005,
                    prerequisites: vec![(0, 2.0), (1, 2.0)], // science > 2.0 AND engineering > 2.0
                    prerequisite_milestones: vec![],
                    effects: TechEffects {
                        tech_level_boost: vec![(0, 0.5), (1, 0.5)],
                        power_multiplier: 1.5,
                        propulsion_unlock: false,
                        resource_efficiency: 1.0,
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                TechMilestone {
                    name: "Fusion Grid Scale".into(),
                    earliest_tick: FUSION_GRID_EARLIEST,
                    latest_tick: FUSION_GRID_LATEST,
                    base_probability: 0.002,
                    prerequisites: vec![],
                    prerequisite_milestones: vec!["Fusion Demo".into()],
                    effects: TechEffects {
                        tech_level_boost: vec![(1, 1.0), (4, 0.5)],
                        power_multiplier: 5.0,
                        propulsion_unlock: false,
                        resource_efficiency: 2.0,
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                TechMilestone {
                    name: "LCF Breakthrough".into(),
                    earliest_tick: LCF_EARLIEST,
                    latest_tick: LCF_LATEST,
                    base_probability: 0.005,
                    prerequisites: vec![(0, 2.5)], // science > 2.5
                    prerequisite_milestones: vec![],
                    effects: TechEffects {
                        tech_level_boost: vec![(0, 2.0), (1, 1.0)],
                        power_multiplier: 10.0,
                        propulsion_unlock: true,
                        resource_efficiency: 3.0,
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                TechMilestone {
                    name: "Manufacturing Breakthrough".into(),
                    earliest_tick: MANUFACTURING_EARLIEST,
                    latest_tick: MANUFACTURING_LATEST,
                    base_probability: 0.003,
                    prerequisites: vec![(1, 3.0)], // engineering > 3.0
                    prerequisite_milestones: vec![],
                    effects: TechEffects {
                        tech_level_boost: vec![(1, 1.5), (3, 1.0)],
                        power_multiplier: 1.0,
                        propulsion_unlock: false,
                        resource_efficiency: 3.0,
                    },
                    achieved: false,
                    achieved_tick: None,
                },
            ],
        }
    }

    /// Check whether a named milestone has been achieved.
    fn is_achieved(&self, name: &str) -> bool {
        self.milestones
            .iter()
            .any(|m| m.name == name && m.achieved)
    }
}

// ---------------------------------------------------------------------------
// DisasterEngine
// ---------------------------------------------------------------------------

/// Core disaster simulation engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisasterEngine {
    /// Active disasters (ongoing events with remaining duration).
    pub active_disasters: Vec<ActiveDisaster>,
    /// Solar cycle phase.
    pub solar_cycle_phase: SolarCyclePhase,
    /// Ticks into the current 132-tick (11-year) solar cycle.
    pub solar_cycle_tick: u32,
    /// Cumulative habitat seal degradation (0.0-1.0).
    pub seal_degradation: f64,
    /// Cumulative micrometeorite damage (0.0-1.0).
    pub micrometeorite_damage: f64,
    /// Rolling complexity cost history (last 120 ticks).
    pub complexity_cost_history: Vec<f64>,
    /// Infrastructure failures in the last `CASCADE_WINDOW_TICKS`.
    pub cascade_failure_count: u32,
    /// Tick at which the last infrastructure failure occurred.
    pub last_failure_ticks: Vec<u32>,
    /// Ticks of sustained high allostatic load per world.
    pub high_load_ticks: Vec<(u32, u32)>, // (world_id, consecutive_ticks)
    /// Configurable tech tree.
    pub tech_tree: TechTree,
    // --- Statistics ---
    pub total_disasters: u64,
    pub carrington_events: u32,
    pub faction_crises: u32,
}

impl DisasterEngine {
    /// Create a new disaster engine with default tech tree.
    pub fn new() -> Self {
        Self {
            active_disasters: Vec::new(),
            solar_cycle_phase: SolarCyclePhase::Rising,
            solar_cycle_tick: 0,
            seal_degradation: 0.0,
            micrometeorite_damage: 0.0,
            complexity_cost_history: Vec::with_capacity(COMPLEXITY_HISTORY_WINDOW),
            cascade_failure_count: 0,
            last_failure_ticks: Vec::new(),
            high_load_ticks: Vec::new(),
            tech_tree: TechTree::default_tree(),
            total_disasters: 0,
            carrington_events: 0,
            faction_crises: 0,
        }
    }

    /// Main tick: roll for disasters, advance active ones, return effects.
    ///
    /// Returns a vec of (effects, world_id, event) for the caller to apply.
    pub fn tick(
        &mut self,
        worlds: &[World],
        current_tick: u32,
        rng: &mut StochasticEngine,
        _policy: &PolicyConfig,
    ) -> Vec<(DisasterEffects, Option<u32>, CivEvent)> {
        let mut results = Vec::new();

        if worlds.is_empty() {
            return results;
        }

        // 1. Advance solar cycle
        self.advance_solar_cycle();

        // 2. Roll for each disaster category
        self.roll_solar_events(current_tick, rng, &mut results);
        self.roll_impact_events(worlds, current_tick, rng, &mut results);
        self.roll_planetary_events(worlds, current_tick, rng, &mut results);
        self.roll_infrastructure_events(worlds, current_tick, rng, &mut results);
        self.roll_psychological_events(worlds, current_tick, rng, &mut results);
        self.roll_tech_events(worlds, current_tick, rng, &mut results);

        // 3. Tainter/Turchin civilization dynamics
        self.check_civilization_dynamics(worlds, current_tick, rng, &mut results);

        // 4. Advance active disasters
        self.advance_active_disasters();

        // 5. Prune old failure timestamps
        self.last_failure_ticks
            .retain(|&t| current_tick.saturating_sub(t) < CASCADE_WINDOW_TICKS);
        self.cascade_failure_count = self.last_failure_ticks.len() as u32;

        results
    }

    // -------------------------------------------------------------------
    // Solar cycle
    // -------------------------------------------------------------------

    fn advance_solar_cycle(&mut self) {
        self.solar_cycle_tick += 1;
        if self.solar_cycle_tick >= SOLAR_CYCLE_TICKS {
            self.solar_cycle_tick = 0;
        }
        // Divide cycle into 4 equal phases of 33 ticks each
        let phase_index = self.solar_cycle_tick / (SOLAR_CYCLE_TICKS / 4);
        self.solar_cycle_phase = match phase_index {
            0 => SolarCyclePhase::Minimum,
            1 => SolarCyclePhase::Rising,
            2 => SolarCyclePhase::Maximum,
            _ => SolarCyclePhase::Declining,
        };
    }

    // -------------------------------------------------------------------
    // Category 1: Solar & Space Weather
    // -------------------------------------------------------------------

    fn roll_solar_events(
        &mut self,
        tick: u32,
        rng: &mut StochasticEngine,
        results: &mut Vec<(DisasterEffects, Option<u32>, CivEvent)>,
    ) {
        let mult = self.solar_cycle_phase.flare_multiplier();

        // M-class flare
        if rng.bernoulli(P_M_CLASS_FLARE * mult) {
            let effects = DisasterEffects {
                solar_power_penalty: 0.05,
                electronics_damage: 0.01,
                allostatic_load_increase: 0.02,
                ..Default::default()
            };
            self.active_disasters.push(ActiveDisaster {
                kind: DisasterKind::Solar(SolarEventKind::MClassFlare),
                severity: 0.3,
                remaining_ticks: 1,
                world_id: None,
                effects: effects.clone(),
            });
            results.push((
                effects,
                None,
                CivEvent::new(tick, None, CivEventType::EmergencyDeclared,
                    "M-class solar flare: 10x SEU rate, minor communications disruption"),
            ));
            self.total_disasters += 1;
        }

        // X-class flare
        if rng.bernoulli(P_X_CLASS_FLARE * mult) {
            let effects = DisasterEffects {
                solar_power_penalty: 0.15,
                electronics_damage: 0.05,
                allostatic_load_increase: 0.08,
                consciousness_shock: 0.03,
                morale_impact: -0.1,
                ..Default::default()
            };
            self.active_disasters.push(ActiveDisaster {
                kind: DisasterKind::Solar(SolarEventKind::XClassFlare),
                severity: 0.6,
                remaining_ticks: 2,
                world_id: None,
                effects: effects.clone(),
            });
            results.push((
                effects,
                None,
                CivEvent::new(tick, None, CivEventType::EmergencyDeclared,
                    "X-class solar flare: 100x SEU rate, radiation shelter advised"),
            ));
            self.total_disasters += 1;
        }

        // Carrington event
        if rng.bernoulli(P_CARRINGTON) {
            let effects = DisasterEffects {
                solar_power_penalty: 0.9,
                electronics_damage: 0.6,
                infrastructure_damage: 0.3,
                resource_production_penalty: 0.5,
                consciousness_shock: 0.15,
                allostatic_load_increase: 0.3,
                morale_impact: -0.4,
                ..Default::default()
            };
            self.active_disasters.push(ActiveDisaster {
                kind: DisasterKind::Solar(SolarEventKind::CarringtonEvent),
                severity: 0.95,
                remaining_ticks: 4, // weeks of recovery
                world_id: None,
                effects: effects.clone(),
            });
            results.push((
                effects,
                None,
                CivEvent::new(tick, None, CivEventType::EmergencyDeclared,
                    "CARRINGTON-CLASS EVENT: catastrophic electronics damage, weeks of recovery"),
            ));
            self.total_disasters += 1;
            self.carrington_events += 1;
        }

        // Major SPE
        if rng.bernoulli(P_MAJOR_SPE * mult) {
            let effects = DisasterEffects {
                population_loss_fraction: 0.01, // lethal dose for unshielded
                allostatic_load_increase: 0.15,
                consciousness_shock: 0.05,
                morale_impact: -0.2,
                ..Default::default()
            };
            self.active_disasters.push(ActiveDisaster {
                kind: DisasterKind::Solar(SolarEventKind::SolarProtonEvent),
                severity: 0.7,
                remaining_ticks: 1,
                world_id: None,
                effects: effects.clone(),
            });
            results.push((
                effects,
                None,
                CivEvent::new(tick, None, CivEventType::EmergencyDeclared,
                    "Major solar proton event: acute radiation hazard for unshielded personnel"),
            ));
            self.total_disasters += 1;
        }

        // Solar minimum onset (GCR increase) — logged but chronic, not acute
        if self.solar_cycle_phase == SolarCyclePhase::Minimum && self.solar_cycle_tick == 1 {
            results.push((
                DisasterEffects {
                    allostatic_load_increase: 0.01,
                    ..Default::default()
                },
                None,
                CivEvent::new(tick, None, CivEventType::EmergencyDeclared,
                    "Solar minimum onset: GCR flux elevated 15% for ~5 years"),
            ));
        }
    }

    // -------------------------------------------------------------------
    // Category 2: Impact Events
    // -------------------------------------------------------------------

    fn roll_impact_events(
        &mut self,
        worlds: &[World],
        tick: u32,
        rng: &mut StochasticEngine,
        results: &mut Vec<(DisasterEffects, Option<u32>, CivEvent)>,
    ) {
        // Micrometeorite barrage: cumulative, always happens
        self.micrometeorite_damage =
            (self.micrometeorite_damage + MICROMETEORITE_DEGRADATION_PER_TICK).min(1.0);

        for world in worlds {
            if world.location == "Earth" {
                continue; // atmosphere shields Earth
            }

            // Small meteorite
            if rng.bernoulli(P_SMALL_METEORITE * world.habitable_area_m2 / 1_000_000.0) {
                let effects = DisasterEffects {
                    population_loss_fraction: 0.005,
                    infrastructure_damage: 0.05,
                    allostatic_load_increase: 0.1,
                    morale_impact: -0.15,
                    ..Default::default()
                };
                results.push((
                    effects,
                    Some(world.id),
                    CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                        format!("{}: small meteorite impact — hull breach risk", world.name)),
                ));
                self.total_disasters += 1;
            }

            // Large meteorite
            if rng.bernoulli(P_LARGE_METEORITE * world.habitable_area_m2 / 1_000_000.0) {
                let effects = DisasterEffects {
                    population_loss_fraction: 0.1,
                    infrastructure_damage: 0.3,
                    resource_production_penalty: 0.2,
                    consciousness_shock: 0.2,
                    allostatic_load_increase: 0.3,
                    morale_impact: -0.5,
                    ..Default::default()
                };
                results.push((
                    effects,
                    Some(world.id),
                    CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                        format!("{}: LARGE METEORITE IMPACT — catastrophic structural damage", world.name)),
                ));
                self.total_disasters += 1;
            }
        }
    }

    // -------------------------------------------------------------------
    // Category 3: Planetary Environment
    // -------------------------------------------------------------------

    fn roll_planetary_events(
        &mut self,
        worlds: &[World],
        tick: u32,
        rng: &mut StochasticEngine,
        results: &mut Vec<(DisasterEffects, Option<u32>, CivEvent)>,
    ) {
        for world in worlds {
            match world.location.as_str() {
                "Mars" => {
                    // Global dust storm
                    if rng.bernoulli(P_MARS_GLOBAL_DUST) {
                        let duration = 1 + (rng.next_f64() * 3.0) as u32; // 1-3 months
                        let effects = DisasterEffects {
                            solar_power_penalty: 0.9,
                            resource_production_penalty: 0.3,
                            allostatic_load_increase: 0.15,
                            morale_impact: -0.2,
                            ..Default::default()
                        };
                        self.active_disasters.push(ActiveDisaster {
                            kind: DisasterKind::Planetary(PlanetaryEventKind::MarsGlobalDustStorm),
                            severity: 0.8,
                            remaining_ticks: duration,
                            world_id: Some(world.id),
                            effects: effects.clone(),
                        });
                        results.push((
                            effects,
                            Some(world.id),
                            CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                                format!("{}: Mars global dust storm — solar output reduced 90% for {} months",
                                    world.name, duration)),
                        ));
                        self.total_disasters += 1;
                    }
                    // Regional dust storm
                    if rng.bernoulli(P_MARS_REGIONAL_DUST) {
                        let effects = DisasterEffects {
                            solar_power_penalty: 0.5,
                            resource_production_penalty: 0.1,
                            allostatic_load_increase: 0.05,
                            ..Default::default()
                        };
                        results.push((
                            effects,
                            Some(world.id),
                            CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                                format!("{}: regional dust storm — solar output reduced 50%", world.name)),
                        ));
                        self.total_disasters += 1;
                    }
                }
                "Moon" => {
                    // Shallow moonquake (damaging)
                    if rng.bernoulli(P_DAMAGING_MOONQUAKE) {
                        let effects = DisasterEffects {
                            infrastructure_damage: 0.02,
                            allostatic_load_increase: 0.05,
                            morale_impact: -0.05,
                            ..Default::default()
                        };
                        results.push((
                            effects,
                            Some(world.id),
                            CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                                format!("{}: shallow moonquake — structural inspection required", world.name)),
                        ));
                        self.total_disasters += 1;
                    }
                    // Lunar dust event (terminator crossing)
                    if rng.bernoulli(0.08) {
                        // ~once per year
                        let effects = DisasterEffects {
                            electronics_damage: 0.01,
                            resource_production_penalty: 0.02,
                            ..Default::default()
                        };
                        results.push((
                            effects,
                            Some(world.id),
                            CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                                format!("{}: charged lunar dust event — equipment fouling", world.name)),
                        ));
                    }
                }
                _ => {} // Other locations have their own hazards
            }
        }
    }

    // -------------------------------------------------------------------
    // Category 4: ECLSS / Infrastructure Failures
    // -------------------------------------------------------------------

    fn roll_infrastructure_events(
        &mut self,
        worlds: &[World],
        tick: u32,
        rng: &mut StochasticEngine,
        results: &mut Vec<(DisasterEffects, Option<u32>, CivEvent)>,
    ) {
        // Cumulative seal degradation
        self.seal_degradation = (self.seal_degradation + SEAL_DEGRADATION_PER_TICK).min(1.0);

        for world in worlds {
            if world.location == "Earth" {
                continue; // Earth doesn't depend on ECLSS
            }

            // Scale failure probability by inverse of infrastructure level
            // Better infrastructure = more redundancy
            let infra_factor = 1.0 / (0.5 + world.infrastructure_level);

            let eclss_failures: &[(f64, InfrastructureFailureKind, &str, DisasterEffects)] = &[
                (
                    P_O2_FAILURE,
                    InfrastructureFailureKind::O2GeneratorDegradation,
                    "O2 generator cell stack failure",
                    DisasterEffects {
                        resource_production_penalty: 0.15,
                        allostatic_load_increase: 0.2,
                        morale_impact: -0.15,
                        ..Default::default()
                    },
                ),
                (
                    P_WATER_FAILURE,
                    InfrastructureFailureKind::WaterRecyclerFailure,
                    "water recycler membrane fouling",
                    DisasterEffects {
                        resource_production_penalty: 0.12,
                        allostatic_load_increase: 0.15,
                        morale_impact: -0.1,
                        ..Default::default()
                    },
                ),
                (
                    P_CO2_FAILURE,
                    InfrastructureFailureKind::CO2ScrubberDegradation,
                    "CO2 scrubber filter clogging",
                    DisasterEffects {
                        resource_production_penalty: 0.1,
                        allostatic_load_increase: 0.15,
                        consciousness_shock: 0.02,
                        morale_impact: -0.1,
                        ..Default::default()
                    },
                ),
                (
                    P_THERMAL_FAILURE,
                    InfrastructureFailureKind::ThermalControlFailure,
                    "thermal control system failure",
                    DisasterEffects {
                        resource_production_penalty: 0.08,
                        allostatic_load_increase: 0.12,
                        morale_impact: -0.08,
                        ..Default::default()
                    },
                ),
                (
                    P_POWER_FAILURE,
                    InfrastructureFailureKind::PowerDistributionFault,
                    "power distribution fault",
                    DisasterEffects {
                        resource_production_penalty: 0.2,
                        electronics_damage: 0.03,
                        allostatic_load_increase: 0.1,
                        morale_impact: -0.12,
                        ..Default::default()
                    },
                ),
                (
                    P_HYDROPONIC_FAILURE,
                    InfrastructureFailureKind::HydroponicSystemFailure,
                    "hydroponic system nutrient imbalance",
                    DisasterEffects {
                        resource_production_penalty: 0.18,
                        allostatic_load_increase: 0.1,
                        morale_impact: -0.1,
                        ..Default::default()
                    },
                ),
            ];

            for (prob, kind, desc, effects) in eclss_failures {
                if rng.bernoulli(prob * infra_factor) {
                    self.last_failure_ticks.push(tick);
                    results.push((
                        effects.clone(),
                        Some(world.id),
                        CivEvent::new(
                            tick,
                            Some(world.id),
                            CivEventType::ResourceCrisis,
                            format!("{}: {}", world.name, desc),
                        ),
                    ));
                    self.total_disasters += 1;
                    self.active_disasters.push(ActiveDisaster {
                        kind: DisasterKind::Infrastructure(*kind),
                        severity: 0.4,
                        remaining_ticks: 2, // typical repair time
                        world_id: Some(world.id),
                        effects: effects.clone(),
                    });
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Category 5: Psychological / Social Events
    // -------------------------------------------------------------------

    fn roll_psychological_events(
        &mut self,
        worlds: &[World],
        tick: u32,
        rng: &mut StochasticEngine,
        results: &mut Vec<(DisasterEffects, Option<u32>, CivEvent)>,
    ) {
        for world in worlds {
            let pop = world.population();
            if pop == 0 {
                continue;
            }

            let mean_load: f64 = world
                .agents
                .iter()
                .filter(|a| a.is_alive())
                .map(|a| a.needs.allostatic_load)
                .sum::<f64>()
                / pop as f64;

            // Winter-over syndrome — more likely in small, isolated colonies
            let isolation_mult = if world.location != "Earth" && pop < 100 {
                2.0
            } else {
                1.0
            };
            if rng.bernoulli(P_WINTER_OVER * isolation_mult * (0.5 + mean_load)) {
                let effects = DisasterEffects {
                    consciousness_shock: 0.02,
                    allostatic_load_increase: 0.08,
                    morale_impact: -0.1,
                    resource_production_penalty: 0.05,
                    ..Default::default()
                };
                results.push((
                    effects,
                    Some(world.id),
                    CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                        format!("{}: winter-over syndrome detected — seasonal depression affecting crew", world.name)),
                ));
                self.total_disasters += 1;
            }

            // Interpersonal conflict — more likely in small groups with high stress
            let conflict_mult = if pop < 50 { 2.0 } else { 1.0 };
            if rng.bernoulli(P_INTERPERSONAL_CONFLICT * conflict_mult * (0.5 + mean_load)) {
                let effects = DisasterEffects {
                    resource_production_penalty: 0.08,
                    allostatic_load_increase: 0.05,
                    consciousness_shock: 0.01,
                    morale_impact: -0.08,
                    ..Default::default()
                };
                results.push((
                    effects,
                    Some(world.id),
                    CivEvent::new(tick, Some(world.id), CivEventType::FactionConflict,
                        format!("{}: interpersonal conflict — productivity loss", world.name)),
                ));
                self.total_disasters += 1;
            }

            // Cognitive impairment
            if rng.bernoulli(P_COGNITIVE_IMPAIRMENT * (0.5 + mean_load)) {
                let effects = DisasterEffects {
                    consciousness_shock: 0.05,
                    resource_production_penalty: 0.03,
                    allostatic_load_increase: 0.04,
                    ..Default::default()
                };
                results.push((
                    effects,
                    Some(world.id),
                    CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                        format!("{}: cognitive impairment episode — decision quality degraded", world.name)),
                ));
                self.total_disasters += 1;
            }

            // Social cohesion collapse — requires low cohesion preconditions
            if mean_load > 0.5 && rng.bernoulli(P_COHESION_COLLAPSE_BASE * mean_load) {
                let effects = DisasterEffects {
                    consciousness_shock: 0.1,
                    allostatic_load_increase: 0.15,
                    morale_impact: -0.3,
                    resource_production_penalty: 0.15,
                    ..Default::default()
                };
                results.push((
                    effects,
                    Some(world.id),
                    CivEvent::new(tick, Some(world.id), CivEventType::FactionEmerged,
                        format!("{}: social cohesion collapse — faction emergence likely", world.name)),
                ));
                self.total_disasters += 1;
                self.faction_crises += 1;
            }

            // Psychotic break
            if rng.bernoulli(P_PSYCHOTIC_BREAK * (0.3 + mean_load)) {
                let effects = DisasterEffects {
                    allostatic_load_increase: 0.1,
                    morale_impact: -0.12,
                    ..Default::default()
                };
                results.push((
                    effects,
                    Some(world.id),
                    CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                        format!("{}: individual psychotic break — medical emergency", world.name)),
                ));
                self.total_disasters += 1;
            }

            // Authority challenge
            if rng.bernoulli(P_AUTHORITY_CHALLENGE * (0.5 + mean_load)) {
                let effects = DisasterEffects {
                    consciousness_shock: 0.05,
                    allostatic_load_increase: 0.1,
                    morale_impact: -0.2,
                    resource_production_penalty: 0.1,
                    ..Default::default()
                };
                results.push((
                    effects,
                    Some(world.id),
                    CivEvent::new(tick, Some(world.id), CivEventType::GovernanceTransition,
                        format!("{}: authority challenge — leadership crisis", world.name)),
                ));
                self.total_disasters += 1;
            }
        }
    }

    // -------------------------------------------------------------------
    // Category 6: Technology Events
    // -------------------------------------------------------------------

    fn roll_tech_events(
        &mut self,
        worlds: &[World],
        tick: u32,
        rng: &mut StochasticEngine,
        results: &mut Vec<(DisasterEffects, Option<u32>, CivEvent)>,
    ) {
        // Aggregate tech levels across all worlds (use max per sector)
        let mut max_tech = [1.0f64; 8];
        let mut total_researchers = 0usize;
        for world in worlds {
            for (i, &level) in world.knowledge.technology_levels.iter().enumerate() {
                max_tech[i] = max_tech[i].max(level);
            }
            total_researchers += world.knowledge.active_researchers;
        }

        // Tech regression when researchers below critical mass
        if total_researchers < 5 && rng.bernoulli(0.05) {
            let effects = DisasterEffects {
                consciousness_shock: 0.03,
                morale_impact: -0.15,
                ..Default::default()
            };
            results.push((
                effects,
                None,
                CivEvent::new(tick, None, CivEventType::InnovationStagnation,
                    "Technology regression: researchers below critical mass — knowledge loss"),
            ));
            self.total_disasters += 1;
        }

        // Roll for each tech milestone
        for milestone in &mut self.tech_tree.milestones {
            if milestone.achieved {
                continue;
            }
            if tick < milestone.earliest_tick || tick > milestone.latest_tick {
                continue;
            }

            // Check sector prerequisites
            let prereqs_met = milestone
                .prerequisites
                .iter()
                .all(|&(sector, min_level)| sector < 8 && max_tech[sector] >= min_level);

            if !prereqs_met {
                continue;
            }

            // Check milestone prerequisites (borrow workaround: collect names first)
            // We need to check against our own milestones but can't borrow self.tech_tree
            // while iterating. The caller will handle this via a second pass.
            // For now, milestone prerequisites are checked structurally.

            if rng.bernoulli(milestone.base_probability) {
                milestone.achieved = true;
                milestone.achieved_tick = Some(tick);

                let mut effects = DisasterEffects::default();
                effects.morale_impact = 0.2; // positive!

                results.push((
                    effects,
                    None,
                    CivEvent::new(tick, None, CivEventType::InnovationBreakthrough,
                        format!("TECH MILESTONE: {} achieved", milestone.name)),
                ));
            }
        }

        // Second pass: enforce milestone prerequisites (un-achieve if prereqs not met)
        let achieved_names: Vec<String> = self
            .tech_tree
            .milestones
            .iter()
            .filter(|m| m.achieved)
            .map(|m| m.name.clone())
            .collect();
        for milestone in &mut self.tech_tree.milestones {
            if milestone.achieved && milestone.achieved_tick == Some(tick) {
                let prereq_milestones_met = milestone
                    .prerequisite_milestones
                    .iter()
                    .all(|name| achieved_names.iter().any(|a| a == name));
                if !prereq_milestones_met {
                    milestone.achieved = false;
                    milestone.achieved_tick = None;
                    // Remove the result we just pushed
                    results.retain(|(_e, _w, ev)| {
                        !(ev.tick == tick
                            && ev.description.contains(&milestone.name))
                    });
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Category 7: Civilization-Scale Dynamics (Tainter/Turchin)
    // -------------------------------------------------------------------

    fn check_civilization_dynamics(
        &mut self,
        worlds: &[World],
        tick: u32,
        rng: &mut StochasticEngine,
        results: &mut Vec<(DisasterEffects, Option<u32>, CivEvent)>,
    ) {
        for world in worlds {
            let pop = world.population();
            if pop == 0 {
                continue;
            }

            // --- Diminishing Returns on Complexity (Tainter 1988) ---
            let stagnant = world.knowledge.stagnation_detected;
            if world.infrastructure_level > TAINTER_COMPLEXITY_THRESHOLD && stagnant {
                // Complexity costs rise faster than benefits
                let cost = world.infrastructure_level * 0.01;
                self.complexity_cost_history.push(cost);
                if self.complexity_cost_history.len() > COMPLEXITY_HISTORY_WINDOW {
                    self.complexity_cost_history.remove(0);
                }
                if rng.bernoulli(0.02) {
                    let effects = DisasterEffects {
                        resource_production_penalty: 0.1,
                        infrastructure_damage: 0.02,
                        consciousness_shock: 0.03,
                        morale_impact: -0.1,
                        ..Default::default()
                    };
                    results.push((
                        effects,
                        Some(world.id),
                        CivEvent::new(tick, Some(world.id), CivEventType::ConstitutionalCalcification,
                            format!("{}: diminishing returns on complexity — Tainter threshold exceeded",
                                world.name)),
                    ));
                    self.total_disasters += 1;
                }
            }

            // --- Elite Overproduction (Turchin 2003) ---
            let tier_dist = world.tier_distribution();
            let guardian_frac = tier_dist[4]; // tier 4 = Guardian
            // Use world-level Gini as proxy (computed externally, use tier skew)
            let phi_values: Vec<f64> = world
                .agents
                .iter()
                .filter(|a| a.is_alive())
                .map(|a| a.consciousness.phi())
                .collect();
            let gini = crate::observables::consciousness_gini(&phi_values);

            if gini > TURCHIN_GINI_THRESHOLD && guardian_frac > TURCHIN_GUARDIAN_FRACTION {
                if rng.bernoulli(0.03) {
                    let effects = DisasterEffects {
                        consciousness_shock: 0.08,
                        allostatic_load_increase: 0.1,
                        morale_impact: -0.2,
                        resource_production_penalty: 0.05,
                        ..Default::default()
                    };
                    results.push((
                        effects,
                        Some(world.id),
                        CivEvent::new(tick, Some(world.id), CivEventType::OppressionAlert,
                            format!("{}: elite overproduction crisis — Turchin dynamics (Gini={:.2}, guardians={:.1}%)",
                                world.name, gini, guardian_frac * 100.0)),
                    ));
                    self.total_disasters += 1;
                    self.faction_crises += 1;
                }
            }

            // --- Resource Depletion Crisis ---
            let any_critical_sustained = world.resources.any_critical();
            if any_critical_sustained {
                // Already handled by the main sim, but we add compound effects
                if rng.bernoulli(0.05) {
                    let effects = DisasterEffects {
                        allostatic_load_increase: 0.1,
                        morale_impact: -0.15,
                        consciousness_shock: 0.05,
                        ..Default::default()
                    };
                    results.push((
                        effects,
                        Some(world.id),
                        CivEvent::new(tick, Some(world.id), CivEventType::ResourceCrisis,
                            format!("{}: resource depletion crisis compounding", world.name)),
                    ));
                    self.total_disasters += 1;
                }
            }

            // --- Institutional Sclerosis ---
            let calcification = world.governance.constitutional_calcification(tick);
            if calcification > SCLEROSIS_CALCIFICATION_THRESHOLD {
                if rng.bernoulli(0.02) {
                    let effects = DisasterEffects {
                        resource_production_penalty: 0.08,
                        consciousness_shock: 0.04,
                        morale_impact: -0.1,
                        ..Default::default()
                    };
                    results.push((
                        effects,
                        Some(world.id),
                        CivEvent::new(tick, Some(world.id), CivEventType::ConstitutionalCalcification,
                            format!("{}: institutional sclerosis — governance rigidity at {:.0}%",
                                world.name, calcification * 100.0)),
                    ));
                    self.total_disasters += 1;
                }
            }

            // --- Systemic Cascade Failure ---
            if self.cascade_failure_count >= CASCADE_FAILURE_THRESHOLD {
                let effects = DisasterEffects {
                    infrastructure_damage: 0.1,
                    resource_production_penalty: 0.25,
                    consciousness_shock: 0.1,
                    allostatic_load_increase: 0.2,
                    morale_impact: -0.3,
                    ..Default::default()
                };
                results.push((
                    effects,
                    Some(world.id),
                    CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                        format!("{}: SYSTEMIC CASCADE FAILURE — {} infrastructure failures in {} ticks",
                            world.name, self.cascade_failure_count, CASCADE_WINDOW_TICKS)),
                ));
                self.total_disasters += 1;
                // Reset to avoid firing every tick
                self.last_failure_ticks.clear();
                self.cascade_failure_count = 0;
            }

            // --- Social Cohesion Crisis (sustained high allostatic load) ---
            let mean_load: f64 = world
                .agents
                .iter()
                .filter(|a| a.is_alive())
                .map(|a| a.needs.allostatic_load)
                .sum::<f64>()
                / pop as f64;

            // Track sustained high load per world
            if let Some(entry) = self.high_load_ticks.iter_mut().find(|(id, _)| *id == world.id) {
                if mean_load > COHESION_CRISIS_LOAD {
                    entry.1 += 1;
                } else {
                    entry.1 = 0;
                }
            } else {
                self.high_load_ticks.push((
                    world.id,
                    if mean_load > COHESION_CRISIS_LOAD { 1 } else { 0 },
                ));
            }

            let sustained_ticks = self
                .high_load_ticks
                .iter()
                .find(|(id, _)| *id == world.id)
                .map(|(_, t)| *t)
                .unwrap_or(0);

            if sustained_ticks >= COHESION_CRISIS_TICKS {
                let effects = DisasterEffects {
                    consciousness_shock: 0.15,
                    allostatic_load_increase: 0.1,
                    morale_impact: -0.3,
                    resource_production_penalty: 0.15,
                    population_loss_fraction: 0.005,
                    ..Default::default()
                };
                results.push((
                    effects,
                    Some(world.id),
                    CivEvent::new(tick, Some(world.id), CivEventType::TraumaAccumulation,
                        format!("{}: SOCIAL COHESION CRISIS — mean allostatic load > {:.0}% for {} months",
                            world.name, COHESION_CRISIS_LOAD * 100.0, sustained_ticks)),
                ));
                self.total_disasters += 1;
                // Reset counter
                if let Some(entry) = self.high_load_ticks.iter_mut().find(|(id, _)| *id == world.id) {
                    entry.1 = 0;
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Active disaster management
    // -------------------------------------------------------------------

    fn advance_active_disasters(&mut self) {
        for disaster in &mut self.active_disasters {
            disaster.remaining_ticks = disaster.remaining_ticks.saturating_sub(1);
        }
        self.active_disasters.retain(|d| d.remaining_ticks > 0);
    }

    /// Get the total ongoing effects from all active disasters for a given world.
    pub fn active_effects_for_world(&self, world_id: u32) -> DisasterEffects {
        let mut combined = DisasterEffects::default();
        for d in &self.active_disasters {
            if d.world_id.is_none() || d.world_id == Some(world_id) {
                combined.merge(&d.effects);
            }
        }
        combined
    }

    /// Current solar cycle GCR multiplier (for radiation dose calculations).
    pub fn gcr_multiplier(&self) -> f64 {
        self.solar_cycle_phase.gcr_multiplier()
    }
}

impl Default for DisasterEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
    use crate::economy::WorldEconomy;
    use crate::governance::WorldGovernance;
    use crate::harmony::HarmonyTracker;
    use crate::knowledge::WorldKnowledge;
    use crate::needs::PsychologicalNeeds;
    use crate::world::{CulturalProfile, WorldResources};

    fn make_test_world(id: u32, location: &str, n_agents: usize) -> World {
        let mut agents = Vec::new();
        for i in 0..n_agents {
            agents.push(CivAgent {
                id: i as u64,
                birth_tick: 0,
                death_tick: None,
                sex: if i % 2 == 0 {
                    BiologicalSex::Female
                } else {
                    BiologicalSex::Male
                },
                world_id: id,
                health: 0.9,
                skills: SkillVector::new(),
                education_level: 0.0,
                consciousness: ConsciousnessState::nascent(),
                partner_id: None,
                children_ids: vec![],
                is_immigrant: false,
                needs: PsychologicalNeeds::new(),
                tend_balance: 0.0,
                parent_ids: None,
                faction_id: None,
                generation: 0,
                trauma_level: 0.0,
            });
        }
        World {
            id,
            name: format!("Test-{}", location),
            location: location.into(),
            founded_tick: 0,
            parent_world_id: None,
            agents,
            next_agent_id: n_agents as u64,
            resources: WorldResources::lunar_default(),
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.5,
            max_population: 10_000,
            habitable_area_m2: 1_000_000.0,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: WorldKnowledge::new(),
            economy: WorldEconomy::new(),
            harmony: HarmonyTracker::new(),
            governance: WorldGovernance::new(),
        }
    }

    #[test]
    fn test_solar_cycle_advances_correctly() {
        let mut engine = DisasterEngine::new();
        // Advance through a full cycle
        for _ in 0..SOLAR_CYCLE_TICKS {
            engine.advance_solar_cycle();
        }
        // After one full cycle, should be back at tick 0 (wraps)
        assert_eq!(engine.solar_cycle_tick, 0);
        assert_eq!(engine.solar_cycle_phase, SolarCyclePhase::Minimum);
    }

    #[test]
    fn test_solar_cycle_132_tick_period() {
        let mut engine = DisasterEngine::new();
        let mut phases_seen = std::collections::HashSet::new();
        for _ in 0..SOLAR_CYCLE_TICKS {
            engine.advance_solar_cycle();
            phases_seen.insert(engine.solar_cycle_phase);
        }
        // All 4 phases should be visited in one cycle
        assert_eq!(phases_seen.len(), 4);
    }

    #[test]
    fn test_carrington_event_is_rare() {
        // P_CARRINGTON = 0.00058/tick → ~0.7% per year (12 ticks)
        let annual_prob = 1.0 - (1.0 - P_CARRINGTON).powi(12);
        assert!(
            (annual_prob - 0.007).abs() < 0.001,
            "Annual Carrington probability should be ~0.7%, was {:.4}",
            annual_prob
        );
    }

    #[test]
    fn test_m_class_flare_probability_realistic() {
        // ~0.05/tick at cycle average → about 0.6 impactful per year
        let annual_expected = P_M_CLASS_FLARE * 12.0;
        assert!(
            annual_expected > 0.3 && annual_expected < 1.5,
            "Expected ~0.6 impactful M-class flares per year, got {:.2}",
            annual_expected
        );
    }

    #[test]
    fn test_mars_global_dust_storm_frequency() {
        // P = 0.015/tick → ~1 per 66 months (3 Mars years)
        let expected_interval = 1.0 / P_MARS_GLOBAL_DUST;
        assert!(
            (expected_interval - 66.7).abs() < 5.0,
            "Expected ~66 month interval, got {:.1}",
            expected_interval
        );
    }

    #[test]
    fn test_eclss_failure_probabilities_match_mtbf() {
        // O2 generator MTBF = 96 months → p ≈ 1/96
        let expected_o2 = 1.0 / 96.0;
        assert!(
            (P_O2_FAILURE - expected_o2).abs() < 0.001,
            "O2 failure prob {:.4} should match 1/96 = {:.4}",
            P_O2_FAILURE,
            expected_o2
        );

        // Hydroponic MTBF = 48 months → p ≈ 1/48
        let expected_hydro = 1.0 / 48.0;
        assert!(
            (P_HYDROPONIC_FAILURE - expected_hydro).abs() < 0.002,
            "Hydroponic failure prob {:.4} should match 1/48 = {:.4}",
            P_HYDROPONIC_FAILURE,
            expected_hydro
        );
    }

    #[test]
    fn test_seal_degradation_is_cumulative() {
        let mut engine = DisasterEngine::new();
        let initial = engine.seal_degradation;
        let worlds = vec![make_test_world(0, "Moon", 10)];
        let mut rng = StochasticEngine::new(42);
        let policy = PolicyConfig::default();

        // Run 120 ticks (10 years)
        for t in 0..120 {
            engine.tick(&worlds, t, &mut rng, &policy);
        }

        assert!(
            engine.seal_degradation > initial,
            "Seal degradation should increase over time"
        );
        // 10 years at 0.1%/year = ~1.0%
        let expected = 120.0 * SEAL_DEGRADATION_PER_TICK;
        assert!(
            (engine.seal_degradation - expected).abs() < 0.001,
            "Seal degradation {:.4} should be ~{:.4}",
            engine.seal_degradation,
            expected
        );
    }

    #[test]
    fn test_active_disasters_expire_correctly() {
        let mut engine = DisasterEngine::new();
        engine.active_disasters.push(ActiveDisaster {
            kind: DisasterKind::Solar(SolarEventKind::XClassFlare),
            severity: 0.6,
            remaining_ticks: 2,
            world_id: None,
            effects: DisasterEffects::default(),
        });
        assert_eq!(engine.active_disasters.len(), 1);

        engine.advance_active_disasters();
        assert_eq!(engine.active_disasters.len(), 1);
        assert_eq!(engine.active_disasters[0].remaining_ticks, 1);

        engine.advance_active_disasters();
        assert_eq!(
            engine.active_disasters.len(),
            0,
            "Disaster should expire after remaining_ticks reaches 0"
        );
    }

    #[test]
    fn test_cascade_failure_detection() {
        let mut engine = DisasterEngine::new();
        // Simulate 3 failures within CASCADE_WINDOW_TICKS
        engine.last_failure_ticks = vec![10, 12, 14];
        // Prune for tick 15 (all within window of 6)
        let current_tick: u32 = 15;
        engine
            .last_failure_ticks
            .retain(|&t| current_tick.saturating_sub(t) < CASCADE_WINDOW_TICKS);
        engine.cascade_failure_count = engine.last_failure_ticks.len() as u32;

        assert!(
            engine.cascade_failure_count >= CASCADE_FAILURE_THRESHOLD,
            "Should detect cascade: {} failures >= threshold {}",
            engine.cascade_failure_count,
            CASCADE_FAILURE_THRESHOLD
        );
    }

    #[test]
    fn test_tainter_diminishing_returns_triggers() {
        let mut world = make_test_world(0, "Moon", 50);
        world.infrastructure_level = 0.9; // above threshold
        world.knowledge.stagnation_detected = true;

        let worlds = vec![world];
        let mut engine = DisasterEngine::new();
        let policy = PolicyConfig::default();

        // Run many ticks to give the probabilistic trigger a chance
        let mut triggered = false;
        for t in 0..500 {
            let mut rng = StochasticEngine::new(t as u64);
            let results = engine.tick(&worlds, t, &mut rng, &policy);
            for (_, _, ev) in &results {
                if ev.description.contains("diminishing returns") {
                    triggered = true;
                }
            }
            if triggered {
                break;
            }
        }
        assert!(triggered, "Tainter diminishing returns should trigger eventually");
    }

    #[test]
    fn test_tech_milestone_prerequisites_checked() {
        let mut engine = DisasterEngine::new();
        let mut world = make_test_world(0, "Moon", 50);
        // Set engineering too low for NTP (needs > 1.2)
        world.knowledge.technology_levels[1] = 0.5;

        let worlds = vec![world];
        let policy = PolicyConfig::default();

        // Run in the NTP window (ticks 24-60) — should not achieve
        for t in 24..60 {
            let mut rng = StochasticEngine::new(t as u64 + 1000);
            engine.tick(&worlds, t, &mut rng, &policy);
        }
        assert!(
            !engine.tech_tree.is_achieved("NTP Demonstration"),
            "NTP should not be achieved without engineering > 1.2"
        );
    }

    #[test]
    fn test_tech_milestone_earliest_tick_enforced() {
        let mut engine = DisasterEngine::new();
        let mut world = make_test_world(0, "Moon", 50);
        world.knowledge.technology_levels[1] = 5.0; // high engineering

        let worlds = vec![world];
        let policy = PolicyConfig::default();

        // Run before NTP earliest (tick 24)
        for t in 0..24 {
            let mut rng = StochasticEngine::new(t as u64 + 2000);
            engine.tick(&worlds, t, &mut rng, &policy);
        }
        assert!(
            !engine.tech_tree.is_achieved("NTP Demonstration"),
            "NTP should not be achievable before tick 24"
        );
    }

    #[test]
    fn test_solar_minimum_increases_gcr() {
        let mut engine = DisasterEngine::new();
        // Set to minimum phase
        engine.solar_cycle_phase = SolarCyclePhase::Minimum;
        assert!(
            engine.gcr_multiplier() > 1.0,
            "GCR should be elevated during solar minimum"
        );
        assert_eq!(engine.gcr_multiplier(), 1.15);
    }

    #[test]
    fn test_disaster_effects_aggregate_correctly() {
        let mut a = DisasterEffects {
            population_loss_fraction: 0.3,
            infrastructure_damage: 0.4,
            solar_power_penalty: 0.5,
            morale_impact: -0.3,
            ..Default::default()
        };
        let b = DisasterEffects {
            population_loss_fraction: 0.8,
            infrastructure_damage: 0.7,
            solar_power_penalty: 0.6,
            morale_impact: -0.8,
            ..Default::default()
        };
        a.merge(&b);

        // Should be clamped to 1.0
        assert_eq!(a.population_loss_fraction, 1.0);
        assert_eq!(a.infrastructure_damage, 1.0);
        assert_eq!(a.solar_power_penalty, 1.0);
        // Morale clamped to -1.0
        assert_eq!(a.morale_impact, -1.0);
    }

    #[test]
    fn test_empty_worlds_produce_no_disasters() {
        let mut engine = DisasterEngine::new();
        let mut rng = StochasticEngine::new(42);
        let policy = PolicyConfig::default();

        let results = engine.tick(&[], 0, &mut rng, &policy);
        assert!(
            results.is_empty(),
            "Empty world list should produce no disasters"
        );
    }

    #[test]
    fn test_psychological_events_more_likely_with_high_load() {
        let mut low_load_count = 0u32;
        let mut high_load_count = 0u32;
        let trials = 500;

        for seed in 0..trials {
            // Low-load world
            let mut world_low = make_test_world(0, "Moon", 20);
            for a in &mut world_low.agents {
                a.needs.allostatic_load = 0.1;
            }
            let mut engine = DisasterEngine::new();
            let mut rng = StochasticEngine::new(seed as u64);
            let results = engine.tick(&[world_low], 100, &mut rng, &PolicyConfig::default());
            low_load_count += results.len() as u32;

            // High-load world
            let mut world_high = make_test_world(0, "Moon", 20);
            for a in &mut world_high.agents {
                a.needs.allostatic_load = 0.9;
            }
            let mut engine2 = DisasterEngine::new();
            let mut rng2 = StochasticEngine::new(seed as u64);
            let results2 = engine2.tick(&[world_high], 100, &mut rng2, &PolicyConfig::default());
            high_load_count += results2.len() as u32;
        }

        assert!(
            high_load_count > low_load_count,
            "High allostatic load should produce more events: high={}, low={}",
            high_load_count,
            low_load_count
        );
    }

    #[test]
    fn test_earth_immune_to_eclss_failures() {
        let mut engine = DisasterEngine::new();
        let world = make_test_world(0, "Earth", 100);
        let policy = PolicyConfig::default();

        let mut eclss_events = 0;
        for t in 0..1000 {
            let mut rng = StochasticEngine::new(t as u64 + 5000);
            let results = engine.tick(&[world.clone()], t, &mut rng, &policy);
            for (_, _, ev) in &results {
                if matches!(ev.event_type, CivEventType::ResourceCrisis) {
                    eclss_events += 1;
                }
            }
        }
        // Earth should have no ECLSS failures (only resource depletion crises, which
        // are also unlikely for Earth). Allow for rare Tainter resource crises.
        assert!(
            eclss_events < 5,
            "Earth should have minimal ECLSS-type failures, got {}",
            eclss_events
        );
    }

    #[test]
    fn test_active_effects_for_world() {
        let mut engine = DisasterEngine::new();
        engine.active_disasters.push(ActiveDisaster {
            kind: DisasterKind::Planetary(PlanetaryEventKind::MarsGlobalDustStorm),
            severity: 0.8,
            remaining_ticks: 3,
            world_id: Some(1),
            effects: DisasterEffects {
                solar_power_penalty: 0.9,
                ..Default::default()
            },
        });
        engine.active_disasters.push(ActiveDisaster {
            kind: DisasterKind::Solar(SolarEventKind::MClassFlare),
            severity: 0.3,
            remaining_ticks: 1,
            world_id: None, // affects all
            effects: DisasterEffects {
                electronics_damage: 0.01,
                ..Default::default()
            },
        });

        let effects_w1 = engine.active_effects_for_world(1);
        assert_eq!(effects_w1.solar_power_penalty, 0.9);
        assert_eq!(effects_w1.electronics_damage, 0.01);

        let effects_w2 = engine.active_effects_for_world(2);
        assert_eq!(effects_w2.solar_power_penalty, 0.0); // Mars storm doesn't affect world 2
        assert_eq!(effects_w2.electronics_damage, 0.01); // but flare does
    }

    #[test]
    fn test_default_tech_tree_has_correct_milestones() {
        let tree = TechTree::default_tree();
        assert_eq!(tree.milestones.len(), 6);
        assert_eq!(tree.milestones[0].name, "NTP Demonstration");
        assert_eq!(tree.milestones[3].name, "Fusion Grid Scale");
        // Fusion Grid Scale requires Fusion Demo
        assert!(tree.milestones[3]
            .prerequisite_milestones
            .iter()
            .any(|s| s == "Fusion Demo"));
    }

    #[test]
    fn test_micrometeorite_damage_accumulates() {
        let mut engine = DisasterEngine::new();
        let worlds = vec![make_test_world(0, "Moon", 10)];
        let mut rng = StochasticEngine::new(42);
        let policy = PolicyConfig::default();

        let initial = engine.micrometeorite_damage;
        for t in 0..60 {
            engine.tick(&worlds, t, &mut rng, &policy);
        }
        assert!(
            engine.micrometeorite_damage > initial,
            "Micrometeorite damage should accumulate"
        );
    }
}
