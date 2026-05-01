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
use crate::empirical;
use crate::events::{CivEvent, CivEventType};
use crate::stochastic::StochasticEngine;
use crate::world::World;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Constants: monthly probabilities derived from published annualized rates
// ---------------------------------------------------------------------------

/// Solar cycle period in ticks (11 years * 12 months).
/// Calibrated: `empirical::SOLAR_CYCLE_MONTHS` (11-year Schwabe cycle).
const SOLAR_CYCLE_TICKS: u32 = empirical::SOLAR_CYCLE_MONTHS;

// Solar & Space Weather — per-tick (monthly) probabilities
/// Impactful M-class flare: ~10/year but only ~10% cause significant disruption.
const P_M_CLASS_FLARE: f64 = 0.05;
/// Damaging X-class flare: ~1/year, ~12% cause ground-level effects.
const P_X_CLASS_FLARE: f64 = 0.01;
/// Carrington-class event: 0.7%/year (Riley 2012) → ~0.00058/month.
/// Cross-ref: `empirical::SPE_EXTREME_PER_YEAR` (0.195/yr from NOAA GOES catalog).
/// Riley estimate is lower because it filters for Carrington-class specifically.
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
/// Mars global dust storm: ~1 per 3 Mars years (~68 months). Zurek & Martin 1993.
/// Calibrated: `1.0 / empirical::MARS_GLOBAL_STORM_FREQUENCY_EARTH_MONTHS` = ~0.0147.
/// Using 0.015 as a round approximation.
const P_MARS_GLOBAL_DUST: f64 = 1.0 / empirical::MARS_GLOBAL_STORM_FREQUENCY_EARTH_MONTHS;
/// Mars regional dust storm (significant): ~8 per Mars year, ~50% impactful.
const P_MARS_REGIONAL_DUST: f64 = 0.05;
/// Damaging shallow moonquake (M5+): ~5.6/year, ~5% structurally damaging.
/// Calibrated: `empirical::SHALLOW_MOONQUAKES_PER_YEAR / 12.0` = ~0.467/month,
/// then ~5% are structurally damaging → 0.023. Using 0.02 as conservative estimate.
const P_DAMAGING_MOONQUAKE: f64 = empirical::SHALLOW_MOONQUAKES_PER_YEAR / 12.0 * 0.05;

// ECLSS / Infrastructure — p = 1 - e^(-1/MTBF) ≈ 1/MTBF for small values
// All MTBF values from `crate::empirical` (NASA TM-2005-214062, ICES-2019-14).
/// O2 generator: MTBF 96 months.
const P_O2_FAILURE: f64 = 1.0 / empirical::ECLSS_O2_GEN_MTBF_MONTHS;
/// Water recycler: MTBF 60 months.
const P_WATER_FAILURE: f64 = 1.0 / empirical::ECLSS_WATER_MTBF_MONTHS;
/// CO2 scrubber: MTBF 72 months.
const P_CO2_FAILURE: f64 = 1.0 / empirical::ECLSS_CO2_MTBF_MONTHS;
/// Thermal control: MTBF 120 months.
const P_THERMAL_FAILURE: f64 = 1.0 / empirical::ECLSS_THERMAL_MTBF_MONTHS;
/// Power distribution: MTBF 84 months.
const P_POWER_FAILURE: f64 = 0.0119;
/// Seal degradation: cumulative 0.1%/year.
const SEAL_DEGRADATION_PER_TICK: f64 = 0.001 / 12.0;
/// Hydroponic system: MTBF 48 months.
const P_HYDROPONIC_FAILURE: f64 = 0.0208;

// ---------------------------------------------------------------------------
// Outer System: Europa (Jupiter radiation belt at 9.38 Rj)
// ---------------------------------------------------------------------------
// Paranicas et al. (2009), "Europa's Radiation Environment", in Europa
// (Univ. Arizona Press); JPL Europa Clipper environmental design docs.

/// Radiation surge from Jupiter magnetosphere compression. ~2%/month.
const P_EUROPA_RADIATION_SURGE: f64 = 0.02;
/// Tidal quake from 30m peak-to-peak flexing (3.55-day orbital period).
/// Moore & Schubert (2000), Icarus. ~8.5 events/month, ~3% damaging.
const P_EUROPA_TIDAL_QUAKE: f64 = 0.03;
/// Ice shell instability (cryovolcanic diapir). Very rare.
/// Pappalardo et al. (1998), Nature.
const P_EUROPA_ICE_INSTABILITY: f64 = 0.001;
/// Europa solar flux: 3.7% of Earth's (50.3 / 1361 W/m²). Panels useless.
const _EUROPA_SOLAR_FLUX_FRACTION: f64 = 0.037;

// ---------------------------------------------------------------------------
// Outer System: Titan (cryogenic at 93.7 K, 1.467 bar N₂)
// ---------------------------------------------------------------------------
// Fulchignoni et al. (2005), Nature 438:785; Niemann et al. (2010),
// JGR 115:E12006; Turtle et al. (2011), Science 331:1414.

/// Heating failure in cryogenic environment: 2× thermal MTBF penalty.
/// At -179°C, heating loss = colony freeze in hours (uninsulated) to days.
const P_TITAN_HEATING_FAILURE: f64 = P_THERMAL_FAILURE * 2.0;
/// Cryogenic embrittlement: cumulative material fatigue at 94 K.
/// Standard steel DBTT well above 94 K; seals and polymers shatter.
const TITAN_EMBRITTLEMENT_PER_TICK: f64 = 0.001;
/// Major methane rainstorm: ~1 per 15 years equatorial. Turtle et al. 2011.
const P_TITAN_METHANE_STORM: f64 = 0.005;
/// Low-gravity chronic health degradation: 0.14g (1.352 m/s²).
/// Estimated 0.5%/month bone loss interpolated from microgravity data.
const TITAN_LOW_G_LOAD_PER_TICK: f64 = 0.002;
/// Titan solar flux: 1.1% of Earth's at orbit, ~1 W/m² at surface after haze.
const _TITAN_SOLAR_FLUX_FRACTION: f64 = 0.011;

// ---------------------------------------------------------------------------
// Earth geophysics
// ---------------------------------------------------------------------------
// USGS historical seismicity catalog; Mason et al. (2004) VEI analysis.

/// Mega-quake M9.0+: ~1 per 80 years (historical record: 5 since 1900).
const P_MEGA_QUAKE: f64 = 0.001;
/// Supervolcanic eruption VEI 7+: ~1 per 80,000 years.
const P_SUPERVOLCANO: f64 = 0.000_001;

// ---------------------------------------------------------------------------
// Magnetosphere decay
// ---------------------------------------------------------------------------
// IGRF-13 model; Pavon-Carrasco & De Santis (2016).

/// Magnetic field decay: ~5% per century (measured ~9% over last 200 years).
const MAGNETIC_DECAY_PER_TICK: f64 = 0.05 / 1200.0;
/// Laschamp-type excursion probability: ~1 per 8,000 years.
const P_EXCURSION: f64 = 0.000_01;
/// Field strength during excursion: 5% of normal. Laschamp event 41 kya.
const EXCURSION_FIELD_STRENGTH: f64 = 0.05;
/// Excursion duration: ~50 years (short excursion, not full reversal).
const EXCURSION_DURATION_TICKS: u32 = 600;

// ---------------------------------------------------------------------------
// Kessler syndrome (orbital debris cascade)
// ---------------------------------------------------------------------------
// Kessler & Cour-Palais (1978), JGR; Liou & Johnson (2006), Adv. Space Res.;
// Liou (2011), NASA ADR parametric study.

/// Debris density doubling time: ~30 years (360 ticks). Liou 2011.
const KESSLER_DOUBLING_TICKS: f64 = 360.0;
/// Governance collapse threshold for cascade trigger.
const KESSLER_GOVERNANCE_THRESHOLD: f64 = 0.3;
/// Sustained collapse duration required (5 years).
const KESSLER_COLLAPSE_DURATION: u32 = 60;
/// Probability of cascade initiation once conditions met.
const P_KESSLER_INITIATION: f64 = 0.05;

/// Maximum concurrent new disasters generated per world per tick.
/// Prevents unrealistic event stacking (5+ simultaneous disasters) that can
/// kill colonies through pure RNG rather than systemic failure.
/// The cascade amplification mechanic (1.0 + 0.5 * (count - 2)) still applies
/// to the disasters that do fire.
const MAX_NEW_DISASTERS_PER_WORLD_PER_TICK: usize = 4;

// Psychological Events — per-tick probabilities for confined crews
// Calibrated from `crate::empirical` psychological isolation data.
/// Winter-over syndrome: ~40% prevalence/year in confined crews (Palinkas 2008).
/// Calibrated: `empirical::WINTER_OVER_PREVALENCE / 10.0` → 0.04/month.
const P_WINTER_OVER: f64 = empirical::WINTER_OVER_PREVALENCE / 10.0;
/// Interpersonal conflict (significant): ~2/year in small crews (Sandal 2006).
const P_INTERPERSONAL_CONFLICT: f64 = 0.017;
/// Cognitive impairment episode: documented in Mars-500 (Basner 2014).
const P_COGNITIVE_IMPAIRMENT: f64 = 0.008;
/// Social cohesion collapse: rare, requires low-cohesion preconditions.
const P_COHESION_COLLAPSE_BASE: f64 = 0.002;
/// Psychotic break: <0.1% with screening (Kanas 2015).
/// Calibrated: `empirical::PSYCHOTIC_BREAK_RATE_SCREENED` (0.001) annualized → ~0.0008/month.
const P_PSYCHOTIC_BREAK: f64 = empirical::PSYCHOTIC_BREAK_RATE_SCREENED * 0.8;
/// Authority challenge: extremely rare.
const P_AUTHORITY_CHALLENGE: f64 = 0.001;

// Tainter/Turchin thresholds
//
// Realism I: Historical collapse calibration reference points.
// These thresholds are calibrated to reproduce known civilizational collapse
// timelines when applied to the sim's Earth colony:
//
// | Civilization    | Duration | Primary Collapse Mode     | Sim Mechanism              |
// |----------------|----------|---------------------------|----------------------------|
// | Roman Empire   | ~500 yr  | Complexity + overstretch  | Tainter at infra > 0.8     |
// | Classic Maya   | ~600 yr  | Environmental + elite     | Turchin + ResourceDepletion|
// | Easter Island  | ~400 yr  | Resource depletion alone  | ResourceDepletionCrisis    |
// | Angkor         | ~600 yr  | Infrastructure decay      | InstitutionalSclerosis     |
// | Soviet Union   | ~70 yr   | Elite + institutional     | Turchin + Sclerosis        |
//
// Validation: if Earth-like world collapses in <200yr or >800yr with default
// config, these thresholds need recalibration.
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
// Anchored to `crate::empirical` technology development timelines.
/// NTP demo: `empirical::NTP_DEMO_YEAR` (2027) → 1 year from epoch start.
/// Sim uses year 2 (tick 24) to account for space-program integration overhead.
const NTP_EARLIEST: u32 = 24;
const NTP_LATEST: u32 = 60;
/// Fission surface power: `empirical::FISSION_SURFACE_POWER_YEAR` (2028) → 2 years.
/// Sim uses year 3 (tick 36) for conservative schedule margin.
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

// Extended tech tree milestones (1000-year arc)
const RADIATION_HARDENING_EARLIEST: u32 = 120; // Year 10
const RADIATION_HARDENING_LATEST: u32 = 600; // Year 50
const CRYO_MATERIALS_EARLIEST: u32 = 240; // Year 20
const CRYO_MATERIALS_LATEST: u32 = 960; // Year 80
const CLOSED_LOOP_ECLSS_EARLIEST: u32 = 360; // Year 30
const CLOSED_LOOP_ECLSS_LATEST: u32 = 1200; // Year 100
const ADR_CAPABILITY_EARLIEST: u32 = 600; // Year 50
const ADR_CAPABILITY_LATEST: u32 = 2400; // Year 200
const BIOREGENERATIVE_AG_EARLIEST: u32 = 480; // Year 40
const BIOREGENERATIVE_AG_LATEST: u32 = 1800; // Year 150
const FUSION_DRIVE_EARLIEST: u32 = 1200; // Year 100
const FUSION_DRIVE_LATEST: u32 = 4800; // Year 400
const QUANTUM_COMMS_EARLIEST: u32 = 2400; // Year 200
const QUANTUM_COMMS_LATEST: u32 = 7200; // Year 600
const TERRAFORMING_PRECURSOR_EARLIEST: u32 = 3600; // Year 300
const TERRAFORMING_PRECURSOR_LATEST: u32 = 9600; // Year 800
const INTERSTELLAR_PROBE_EARLIEST: u32 = 6000; // Year 500
const INTERSTELLAR_PROBE_LATEST: u32 = 12000; // Year 1000

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
    // --- Europa (Jupiter system) ---
    /// Jupiter magnetosphere compression → radiation surge. Paranicas 2009.
    EuropaRadiationSurge,
    /// Tidal flexing stress (30m peak-to-peak, 3.55-day cycle). Moore 2000.
    EuropaTidalQuake,
    /// Cryovolcanic/diapir event in ice shell. Pappalardo 1998.
    EuropaIceShellInstability,
    // --- Titan (Saturn system) ---
    /// Thermal control failure in cryogenic environment (-179°C). 2× MTBF.
    TitanHeatingFailure,
    /// Cumulative material fatigue from 94 K thermal cycling.
    TitanCryogenicEmbrittlement,
    /// Major methane rainstorm + flash flooding. Turtle 2011.
    TitanMethaneStorm,
    /// Chronic health degradation at 0.14g. Deterministic per-tick.
    TitanLowGravityHealth,
    // --- Earth geophysics ---
    /// M9.0+ mega-earthquake. ~1 per 80 years. USGS catalog.
    EarthMegaQuake,
    /// Tsunami triggered by mega-quake (50% co-occurrence).
    EarthMegaTsunami,
    /// VEI 7+ supervolcanic eruption. ~1 per 80,000 years. Global effects.
    EarthSupervolcanicEruption,
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
    /// LEO debris cascade triggered by governance collapse. Kessler 1978.
    KesslerCascade,
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
    /// Fix 10: Apply continuous time phase scaling to disaster effects.
    ///
    /// `event_phase` is in [0.0, 1.0) representing the point in the
    /// colony's rotation/day-cycle when the event hits.
    ///
    /// - Day-side (phase < 0.5): full damage
    /// - Night-side (phase >= 0.5): 30% damage for solar events (shielded by body)
    /// - Night shift (phase >= 0.75): slower response → 1.2× infrastructure damage
    ///
    /// `is_solar` indicates whether this is a solar/radiation event.
    pub fn apply_event_phase(&mut self, event_phase: f64, is_solar: bool) {
        if is_solar && event_phase >= 0.5 {
            // Night-side: solar events do only 30% damage
            self.population_loss_fraction *= 0.3;
            self.electronics_damage *= 0.3;
            self.consciousness_shock *= 0.3;
            self.solar_power_penalty *= 0.3;
        }

        if event_phase >= 0.75 {
            // Night shift: slower emergency response → more infrastructure damage
            self.infrastructure_damage *= 1.2;
            self.resource_production_penalty *= 1.1;
        }
    }

    /// Combine two effect sets (additive, clamped).
    pub fn merge(&mut self, other: &DisasterEffects) {
        self.population_loss_fraction =
            (self.population_loss_fraction + other.population_loss_fraction).min(1.0);
        self.infrastructure_damage =
            (self.infrastructure_damage + other.infrastructure_damage).min(1.0);
        self.resource_production_penalty =
            (self.resource_production_penalty + other.resource_production_penalty).min(1.0);
        self.solar_power_penalty = (self.solar_power_penalty + other.solar_power_penalty).min(1.0);
        self.consciousness_shock = (self.consciousness_shock + other.consciousness_shock).min(1.0);
        self.allostatic_load_increase =
            (self.allostatic_load_increase + other.allostatic_load_increase).min(1.0);
        self.electronics_damage = (self.electronics_damage + other.electronics_damage).min(1.0);
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
    ///
    /// `base_probability` values are calibrated from expert surveys and program
    /// timelines (2025 data): NASA FSP contracts, ITER/DEMO schedules, DARPA DRACO,
    /// ESA MELiSSA, ClearSpace-1, EDEN ISS, Metaculus community predictions.
    /// Key calibration: Fission 0.025 (mature), Fusion Drive 0.0005 (speculative),
    /// Bioregenerative Ag 0.003 (lighting energy dominates), Cryogenic Materials
    /// 0.012 (well-understood, Dragonfly validates).
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
                    base_probability: 0.025,
                    // Lowered from 1.5 to 1.2 — fission is mature tech (TRL 5-6),
                    // the barrier is deployment logistics not fundamental science.
                    // KRUSTY tested 2018, NASA FSP contracts awarded 2022.
                    prerequisites: vec![(0, 1.2)], // engineering (sector 0) > 1.2
                    prerequisite_milestones: vec![],
                    effects: TechEffects {
                        tech_level_boost: vec![(0, 0.5), (4, 0.2)], // boost engineering + science
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
                    base_probability: 0.004,
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
                    base_probability: 0.0015,
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
                    base_probability: 0.002,
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
                // === Extended tech tree (1000-year arc) ===
                TechMilestone {
                    name: "Radiation Hardening".into(),
                    earliest_tick: RADIATION_HARDENING_EARLIEST,
                    latest_tick: RADIATION_HARDENING_LATEST,
                    base_probability: 0.006,
                    prerequisites: vec![(0, 1.5), (4, 1.3)], // engineering + science
                    prerequisite_milestones: vec!["Fission Surface Power".into()],
                    effects: TechEffects {
                        tech_level_boost: vec![(0, 0.3), (4, 0.2)],
                        power_multiplier: 1.0,
                        propulsion_unlock: false,
                        resource_efficiency: 1.3, // less electronics replacement
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                TechMilestone {
                    name: "Cryogenic Materials".into(),
                    earliest_tick: CRYO_MATERIALS_EARLIEST,
                    latest_tick: CRYO_MATERIALS_LATEST,
                    base_probability: 0.012,
                    prerequisites: vec![(0, 1.8), (4, 1.5)],
                    prerequisite_milestones: vec![],
                    effects: TechEffects {
                        tech_level_boost: vec![(0, 0.4)],
                        power_multiplier: 1.0,
                        propulsion_unlock: false,
                        resource_efficiency: 1.5, // better seals and structures
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                TechMilestone {
                    name: "Closed-Loop ECLSS".into(),
                    earliest_tick: CLOSED_LOOP_ECLSS_EARLIEST,
                    latest_tick: CLOSED_LOOP_ECLSS_LATEST,
                    base_probability: 0.008,
                    prerequisites: vec![(0, 2.0), (1, 1.5)], // engineering + agriculture
                    prerequisite_milestones: vec![],
                    effects: TechEffects {
                        tech_level_boost: vec![(0, 0.3), (1, 0.5)],
                        power_multiplier: 1.0,
                        propulsion_unlock: false,
                        resource_efficiency: 2.0, // halves resource dependency
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                TechMilestone {
                    name: "ADR Capability".into(),
                    earliest_tick: ADR_CAPABILITY_EARLIEST,
                    latest_tick: ADR_CAPABILITY_LATEST,
                    base_probability: 0.004,
                    prerequisites: vec![(0, 2.5), (7, 1.5)], // engineering + logistics
                    prerequisite_milestones: vec!["Manufacturing Breakthrough".into()],
                    effects: TechEffects {
                        tech_level_boost: vec![(0, 0.2), (7, 0.3)],
                        power_multiplier: 1.0,
                        propulsion_unlock: false,
                        resource_efficiency: 1.0,
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                TechMilestone {
                    name: "Bioregenerative Agriculture".into(),
                    earliest_tick: BIOREGENERATIVE_AG_EARLIEST,
                    latest_tick: BIOREGENERATIVE_AG_LATEST,
                    base_probability: 0.003,
                    prerequisites: vec![(1, 2.0), (4, 1.8)], // agriculture + science
                    prerequisite_milestones: vec![],
                    effects: TechEffects {
                        tech_level_boost: vec![(1, 1.0), (4, 0.3)],
                        power_multiplier: 1.0,
                        propulsion_unlock: false,
                        resource_efficiency: 2.5, // near-complete food self-sufficiency
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                TechMilestone {
                    name: "Fusion Drive".into(),
                    earliest_tick: FUSION_DRIVE_EARLIEST,
                    latest_tick: FUSION_DRIVE_LATEST,
                    base_probability: 0.0005,
                    prerequisites: vec![(0, 3.5), (4, 3.0)],
                    prerequisite_milestones: vec!["Fusion Grid Scale".into()],
                    effects: TechEffects {
                        tech_level_boost: vec![(0, 1.0), (7, 1.0)],
                        power_multiplier: 1.0,
                        propulsion_unlock: true, // eliminates transfer windows
                        resource_efficiency: 1.5,
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                TechMilestone {
                    name: "Quantum Communications".into(),
                    earliest_tick: QUANTUM_COMMS_EARLIEST,
                    latest_tick: QUANTUM_COMMS_LATEST,
                    base_probability: 0.003,
                    prerequisites: vec![(4, 3.5)], // science > 3.5
                    prerequisite_milestones: vec!["Fusion Demo".into()],
                    effects: TechEffects {
                        tech_level_boost: vec![(4, 1.0), (5, 0.5)],
                        power_multiplier: 1.0,
                        propulsion_unlock: false,
                        resource_efficiency: 1.0,
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                // Fix 3: Genetic Engineering eliminates inbreeding depression.
                // Enables small colonies to maintain genetic diversity via CRISPR/
                // Yamanaka factor therapies. Realistic timeline: 2060-2200.
                // Rejuvenate Bio (2024): 109% lifespan increase in mice with OSK.
                TechMilestone {
                    name: "Genetic Engineering".into(),
                    earliest_tick: CRYO_MATERIALS_EARLIEST, // Year 20
                    latest_tick: FUSION_DRIVE_LATEST,       // Year 400
                    base_probability: 0.004,
                    prerequisites: vec![(4, 2.5), (2, 2.0)], // science + medicine
                    prerequisite_milestones: vec![],
                    effects: TechEffects {
                        tech_level_boost: vec![(2, 0.8), (4, 0.3)],
                        power_multiplier: 1.0,
                        propulsion_unlock: false,
                        resource_efficiency: 1.0,
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                TechMilestone {
                    name: "Terraforming Precursor".into(),
                    earliest_tick: TERRAFORMING_PRECURSOR_EARLIEST,
                    latest_tick: TERRAFORMING_PRECURSOR_LATEST,
                    base_probability: 0.001,
                    prerequisites: vec![(0, 4.0), (1, 3.0), (4, 4.0)],
                    prerequisite_milestones: vec![
                        "Bioregenerative Agriculture".into(),
                        "Fusion Grid Scale".into(),
                    ],
                    effects: TechEffects {
                        tech_level_boost: vec![(0, 1.5), (1, 1.5), (4, 1.0)],
                        power_multiplier: 1.0,
                        propulsion_unlock: false,
                        resource_efficiency: 5.0,
                    },
                    achieved: false,
                    achieved_tick: None,
                },
                TechMilestone {
                    name: "Interstellar Probe".into(),
                    earliest_tick: INTERSTELLAR_PROBE_EARLIEST,
                    latest_tick: INTERSTELLAR_PROBE_LATEST,
                    base_probability: 0.0002,
                    prerequisites: vec![(0, 5.0), (4, 5.0), (7, 3.0)],
                    prerequisite_milestones: vec![
                        "Fusion Drive".into(),
                        "Quantum Communications".into(),
                    ],
                    effects: TechEffects {
                        tech_level_boost: vec![(0, 2.0), (4, 2.0), (6, 1.0)],
                        power_multiplier: 1.0,
                        propulsion_unlock: true,
                        resource_efficiency: 1.0,
                    },
                    achieved: false,
                    achieved_tick: None,
                },
            ],
        }
    }

    /// Check whether a named milestone has been achieved.
    pub fn is_achieved(&self, name: &str) -> bool {
        self.milestones.iter().any(|m| m.name == name && m.achieved)
    }
}

// ---------------------------------------------------------------------------
// Magnetosphere state (Earth's geomagnetic field)
// ---------------------------------------------------------------------------

/// Earth's magnetic field state — decays over centuries, occasional excursions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagnetosphereState {
    /// Normalized field strength (1.0 = current 2025, decays toward 0).
    pub field_strength: f64,
    /// Whether a Laschamp-type excursion is active.
    pub excursion_active: bool,
    /// Ticks remaining in the current excursion.
    pub excursion_remaining_ticks: u32,
}

impl Default for MagnetosphereState {
    fn default() -> Self {
        Self {
            field_strength: 1.0,
            excursion_active: false,
            excursion_remaining_ticks: 0,
        }
    }
}

impl MagnetosphereState {
    /// Advance magnetosphere state by one tick.
    pub fn tick(&mut self, rng: &mut StochasticEngine) {
        // Secular decay: ~5% per century
        self.field_strength = (self.field_strength - MAGNETIC_DECAY_PER_TICK).max(0.01);

        if self.excursion_active {
            if self.excursion_remaining_ticks > 0 {
                self.excursion_remaining_ticks -= 1;
            } else {
                // Excursion ends — field recovers to pre-excursion level
                self.excursion_active = false;
                // Recovery is partial — field returns to decayed baseline, not 1.0
            }
        } else if rng.bernoulli(P_EXCURSION) {
            self.excursion_active = true;
            self.excursion_remaining_ticks = EXCURSION_DURATION_TICKS;
        }
    }

    /// Effective field strength (drops to 5% during excursion).
    pub fn effective_strength(&self) -> f64 {
        if self.excursion_active {
            self.field_strength * EXCURSION_FIELD_STRENGTH
        } else {
            self.field_strength
        }
    }

    /// Solar event severity multiplier for Earth.
    /// Weaker field → more severe solar events reaching the surface.
    /// At full strength: 1.0. At 5% (excursion): ~4.0.
    pub fn solar_severity_multiplier(&self) -> f64 {
        let eff = self.effective_strength();
        1.0 + 3.0 * (1.0 - eff).powi(2)
    }
}

// ---------------------------------------------------------------------------
// Orbital debris state (Kessler syndrome)
// ---------------------------------------------------------------------------

/// LEO orbital debris density and cascade dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalDebrisState {
    /// Debris density relative to 2025 baseline (1.0 = current).
    pub density_fraction: f64,
    /// Whether a self-sustaining collision cascade is active.
    pub cascade_active: bool,
    /// Tick at which the cascade began.
    pub cascade_start_tick: Option<u32>,
    /// Active debris removal capacity (normalized, 0 = none).
    pub adr_capacity: f64,
    /// LEO access multiplier (1.0 = normal, → 0.0 as density grows).
    pub leo_access_multiplier: f64,
    /// Consecutive ticks of Earth governance below threshold.
    pub governance_collapse_ticks: u32,
}

impl Default for OrbitalDebrisState {
    fn default() -> Self {
        Self {
            density_fraction: 1.0,
            cascade_active: false,
            cascade_start_tick: None,
            adr_capacity: 0.0,
            leo_access_multiplier: 1.0,
            governance_collapse_ticks: 0,
        }
    }
}

impl OrbitalDebrisState {
    /// Advance debris state by one tick.
    pub fn tick(
        &mut self,
        earth_governance_stability: f64,
        has_manufacturing: bool,
        current_tick: u32,
        rng: &mut StochasticEngine,
    ) {
        // Track governance collapse duration
        if earth_governance_stability < KESSLER_GOVERNANCE_THRESHOLD {
            self.governance_collapse_ticks += 1;
        } else {
            self.governance_collapse_ticks = self.governance_collapse_ticks.saturating_sub(2);
        }

        // Cascade trigger
        if !self.cascade_active
            && self.governance_collapse_ticks >= KESSLER_COLLAPSE_DURATION
            && rng.bernoulli(P_KESSLER_INITIATION)
        {
            self.cascade_active = true;
            self.cascade_start_tick = Some(current_tick);
        }

        // Cascade dynamics
        if self.cascade_active {
            // Exponential growth: doubles every 30 years (360 ticks)
            self.density_fraction *= 1.0 + (0.693 / KESSLER_DOUBLING_TICKS);
        } else {
            // Slow natural decay from atmospheric drag (sub-600km)
            self.density_fraction = (self.density_fraction * 0.999).max(1.0);
        }

        // Active debris removal
        if has_manufacturing && earth_governance_stability > 0.5 {
            self.adr_capacity = 1.0; // Full ADR capability
        } else if earth_governance_stability > 0.3 {
            self.adr_capacity = 0.3; // Partial
        } else {
            self.adr_capacity = 0.0;
        }
        self.density_fraction = (self.density_fraction - self.adr_capacity * 0.002).max(1.0);

        // LEO access multiplier: drops as density grows
        // At 1x: 1.0, at 10x: ~0.9, at 100x: ~0.5, at 1000x: ~0.09
        self.leo_access_multiplier =
            (1.0 / (1.0 + (self.density_fraction - 1.0) * 0.01)).clamp(0.0, 1.0);

        // Cascade can be arrested if density drops back to baseline
        if self.cascade_active && self.density_fraction <= 1.5 {
            self.cascade_active = false;
        }
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
    /// Mechanism 5 — Collective Memory Inoculation: disaster kinds the civilization
    /// has survived (remaining_ticks reached 0). Subsequent occurrences of the same
    /// kind have severity reduced by 30% (institutional learning).
    pub survived_disaster_types: HashSet<String>,
    /// Mechanism 1 — Non-Linear Cascade Failures: count of active disasters per world.
    /// When a world has 3+ active disasters, all new effects are amplified.
    pub active_per_world: HashMap<u32, u32>,
    // --- Geophysics & orbital environment ---
    /// Earth's magnetic field state.
    pub magnetosphere: MagnetosphereState,
    /// LEO orbital debris (Kessler syndrome).
    pub orbital_debris: OrbitalDebrisState,
    /// Cumulative Titan embrittlement damage (0.0-1.0).
    pub titan_embrittlement: f64,
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
            survived_disaster_types: HashSet::new(),
            active_per_world: HashMap::new(),
            magnetosphere: MagnetosphereState::default(),
            orbital_debris: OrbitalDebrisState::default(),
            titan_embrittlement: 0.0,
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

        // 1.5. Advance geophysics subsystems
        self.magnetosphere.tick(rng);
        let earth_gov_stability = worlds
            .iter()
            .find(|w| w.location == "Earth")
            .map(|w| w.governance.stability_score)
            .unwrap_or(1.0);
        let has_manufacturing = self.tech_tree.is_achieved("Manufacturing Breakthrough");
        self.orbital_debris
            .tick(earth_gov_stability, has_manufacturing, current_tick, rng);

        // Kessler cascade event generation
        if self.orbital_debris.cascade_active
            && self.orbital_debris.cascade_start_tick == Some(current_tick)
        {
            let effects = DisasterEffects {
                resource_production_penalty: 0.15,
                consciousness_shock: 0.05,
                allostatic_load_increase: 0.1,
                morale_impact: -0.2,
                ..Default::default()
            };
            results.push((
                effects,
                None,
                CivEvent::new(current_tick, None, CivEventType::EmergencyDeclared,
                    format!("KESSLER CASCADE INITIATED — LEO debris density {:.1}x baseline, space access degrading",
                        self.orbital_debris.density_fraction)),
            ));
            self.total_disasters += 1;
        }

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

        // 4.5. Mechanism 1 — Non-Linear Cascade Failures: recompute active disaster
        // counts per world. When a world has 3+ active disasters, all effects from
        // this tick are amplified by 1.0 + 0.5 * (active_count - 2).
        self.active_per_world.clear();
        for d in &self.active_disasters {
            if let Some(wid) = d.world_id {
                *self.active_per_world.entry(wid).or_insert(0) += 1;
            }
        }
        // Also count global disasters (world_id == None) toward all worlds
        let global_count = self
            .active_disasters
            .iter()
            .filter(|d| d.world_id.is_none())
            .count() as u32;
        for world in worlds {
            let entry = self.active_per_world.entry(world.id).or_insert(0);
            *entry += global_count;
        }

        // 5. Prune old failure timestamps
        self.last_failure_ticks
            .retain(|&t| current_tick.saturating_sub(t) < CASCADE_WINDOW_TICKS);
        self.cascade_failure_count = self.last_failure_ticks.len() as u32;

        // Cap new disasters per world per tick to prevent unrealistic stacking.
        // Count how many results target each world and drop excess.
        {
            let mut per_world_count: std::collections::HashMap<u32, usize> =
                std::collections::HashMap::new();
            results.retain(|(_, world_id, _)| {
                if let Some(wid) = world_id {
                    let count = per_world_count.entry(*wid).or_insert(0);
                    *count += 1;
                    *count <= MAX_NEW_DISASTERS_PER_WORLD_PER_TICK
                } else {
                    true // Global disasters (no world_id) are never capped
                }
            });
        }

        // Dead Loop #2 fix: Apply collective memory inoculation to ALL disaster
        // effects. Civilizations that have survived a disaster type before take
        // 30% less damage from subsequent occurrences (Mechanism 5).
        for (effects, _world_id, _event) in &mut results {
            // Infer disaster kind from the event for inoculation lookup.
            // Since effects don't carry kind, we apply a blanket inoculation
            // based on whether ANY of this tick's disaster kinds were survived.
            // This is a conservative approximation — ideally each result would
            // carry its DisasterKind, but the current API doesn't support that.
            let inoc = if !self.survived_disaster_types.is_empty() {
                // Average inoculation across all survived types relevant this tick
                0.85 // 15% average reduction when civilization has disaster memory
            } else {
                1.0
            };
            effects.consciousness_shock *= inoc;
            effects.allostatic_load_increase *= inoc;
            effects.infrastructure_damage *= inoc;
            effects.population_loss_fraction *= inoc;
            effects.resource_production_penalty *= inoc;
        }

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
                CivEvent::new(
                    tick,
                    None,
                    CivEventType::EmergencyDeclared,
                    "M-class solar flare: 10x SEU rate, minor communications disruption",
                ),
            ));
            self.total_disasters += 1;
        }

        // X-class flare
        if rng.bernoulli(P_X_CLASS_FLARE * mult) {
            // Mechanism 2 — Milestone Shielding: smaller reductions for X-class
            // (40% for fission, 20% for manufacturing).
            let mut xclass_severity = 1.0_f64;
            if self.tech_tree.is_achieved("Fission Surface Power") {
                xclass_severity *= 0.6; // 40% reduction
            }
            if self.tech_tree.is_achieved("Manufacturing Breakthrough") {
                xclass_severity *= 0.8; // additional 20% reduction
            }
            xclass_severity *=
                self.inoculation_factor(DisasterKind::Solar(SolarEventKind::XClassFlare));
            let effects = DisasterEffects {
                solar_power_penalty: 0.15 * xclass_severity,
                electronics_damage: 0.05 * xclass_severity,
                allostatic_load_increase: 0.08 * xclass_severity,
                consciousness_shock: 0.03 * xclass_severity,
                morale_impact: -0.1 * xclass_severity,
                ..Default::default()
            };
            self.active_disasters.push(ActiveDisaster {
                kind: DisasterKind::Solar(SolarEventKind::XClassFlare),
                severity: 0.6 * xclass_severity,
                remaining_ticks: 2,
                world_id: None,
                effects: effects.clone(),
            });
            results.push((
                effects,
                None,
                CivEvent::new(tick, None, CivEventType::EmergencyDeclared,
                    format!("X-class solar flare: 100x SEU rate, radiation shelter advised (severity {:.0}%)",
                        xclass_severity * 100.0)),
            ));
            self.total_disasters += 1;
        }

        // Carrington event
        if rng.bernoulli(P_CARRINGTON) {
            // Mechanism 2 — Milestone Shielding (Carrington Defense): fission power
            // provides hardened electronics + nuclear backup (60% reduction), and
            // manufacturing capability enables local rebuild (additional 30% reduction).
            let mut carrington_severity = 1.0_f64;
            // Magnetosphere decay amplifies Carrington severity for Earth.
            // At full field: 1.0×. At excursion (5%): ~4.0×.
            carrington_severity *= self.magnetosphere.solar_severity_multiplier();
            if self.tech_tree.is_achieved("Fission Surface Power") {
                carrington_severity *= 0.4; // 60% reduction
            }
            if self.tech_tree.is_achieved("Manufacturing Breakthrough") {
                carrington_severity *= 0.7; // additional 30% reduction
            }
            // Mechanism 5 — Collective Memory Inoculation
            carrington_severity *=
                self.inoculation_factor(DisasterKind::Solar(SolarEventKind::CarringtonEvent));
            let effects = DisasterEffects {
                solar_power_penalty: 0.9 * carrington_severity,
                electronics_damage: 0.6 * carrington_severity,
                infrastructure_damage: 0.3 * carrington_severity,
                resource_production_penalty: 0.5 * carrington_severity,
                consciousness_shock: 0.15 * carrington_severity,
                allostatic_load_increase: 0.3 * carrington_severity,
                morale_impact: -0.4 * carrington_severity,
                ..Default::default()
            };
            self.active_disasters.push(ActiveDisaster {
                kind: DisasterKind::Solar(SolarEventKind::CarringtonEvent),
                severity: 0.95 * carrington_severity,
                remaining_ticks: 4, // weeks of recovery
                world_id: None,
                effects: effects.clone(),
            });
            results.push((
                effects,
                None,
                CivEvent::new(
                    tick,
                    None,
                    CivEventType::EmergencyDeclared,
                    format!(
                        "CARRINGTON-CLASS EVENT: catastrophic electronics damage (severity {:.0}%)",
                        carrington_severity * 100.0
                    ),
                ),
            ));
            self.total_disasters += 1;
            self.carrington_events += 1;
        }

        // Major SPE
        if rng.bernoulli(P_MAJOR_SPE * mult) {
            // Mechanism 2 — Milestone Shielding: 40% fission, 20% manufacturing.
            let mut spe_severity = 1.0_f64;
            if self.tech_tree.is_achieved("Fission Surface Power") {
                spe_severity *= 0.6; // 40% reduction
            }
            if self.tech_tree.is_achieved("Manufacturing Breakthrough") {
                spe_severity *= 0.8; // additional 20% reduction
            }
            spe_severity *=
                self.inoculation_factor(DisasterKind::Solar(SolarEventKind::SolarProtonEvent));
            let effects = DisasterEffects {
                population_loss_fraction: 0.01 * spe_severity,
                allostatic_load_increase: 0.15 * spe_severity,
                consciousness_shock: 0.05 * spe_severity,
                morale_impact: -0.2 * spe_severity,
                ..Default::default()
            };
            self.active_disasters.push(ActiveDisaster {
                kind: DisasterKind::Solar(SolarEventKind::SolarProtonEvent),
                severity: 0.7 * spe_severity,
                remaining_ticks: 1,
                world_id: None,
                effects: effects.clone(),
            });
            results.push((
                effects,
                None,
                CivEvent::new(
                    tick,
                    None,
                    CivEventType::EmergencyDeclared,
                    format!(
                        "Major solar proton event: acute radiation hazard (severity {:.0}%)",
                        spe_severity * 100.0
                    ),
                ),
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
                CivEvent::new(
                    tick,
                    None,
                    CivEventType::EmergencyDeclared,
                    "Solar minimum onset: GCR flux elevated 15% for ~5 years",
                ),
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
                    CivEvent::new(
                        tick,
                        Some(world.id),
                        CivEventType::EmergencyDeclared,
                        format!("{}: small meteorite impact — hull breach risk", world.name),
                    ),
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
                    CivEvent::new(
                        tick,
                        Some(world.id),
                        CivEventType::EmergencyDeclared,
                        format!(
                            "{}: LARGE METEORITE IMPACT — catastrophic structural damage",
                            world.name
                        ),
                    ),
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
                            CivEvent::new(
                                tick,
                                Some(world.id),
                                CivEventType::EmergencyDeclared,
                                format!(
                                    "{}: regional dust storm — solar output reduced 50%",
                                    world.name
                                ),
                            ),
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
                            CivEvent::new(
                                tick,
                                Some(world.id),
                                CivEventType::EmergencyDeclared,
                                format!(
                                    "{}: shallow moonquake — structural inspection required",
                                    world.name
                                ),
                            ),
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
                            CivEvent::new(
                                tick,
                                Some(world.id),
                                CivEventType::EmergencyDeclared,
                                format!(
                                    "{}: charged lunar dust event — equipment fouling",
                                    world.name
                                ),
                            ),
                        ));
                    }
                }
                "Europa" => {
                    // Jupiter radiation surge — magnetosphere compression event
                    if rng.bernoulli(P_EUROPA_RADIATION_SURGE) {
                        // Subterranean colonies (infrastructure > 0.3 implies buried) get 90% reduction
                        let mut shielding = if world.infrastructure_level > 0.3 {
                            0.1
                        } else {
                            1.0
                        };
                        // Radiation Hardening tech: additional 60% reduction
                        if self.tech_tree.is_achieved("Radiation Hardening") {
                            shielding *= 0.4;
                        }
                        let effects = DisasterEffects {
                            electronics_damage: 0.05 * shielding,
                            population_loss_fraction: 0.005 * shielding,
                            consciousness_shock: 0.1 * shielding,
                            allostatic_load_increase: 0.1,
                            morale_impact: -0.15,
                            ..Default::default()
                        };
                        results.push((
                            effects,
                            Some(world.id),
                            CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                                format!("{}: Jupiter magnetosphere compression — radiation surge (shielding {:.0}%)",
                                    world.name, (1.0 - shielding) * 100.0)),
                        ));
                        self.total_disasters += 1;
                    }

                    // Tidal quake (30m peak-to-peak flexing, 3.55-day cycle)
                    if rng.bernoulli(P_EUROPA_TIDAL_QUAKE) {
                        let damage = 0.01 + rng.next_f64() * 0.02; // 0.01-0.03
                                                                   // Better infrastructure flexes with the ice
                        let flex_factor = 1.0 - world.infrastructure_level * 0.5;
                        let effects = DisasterEffects {
                            infrastructure_damage: damage * flex_factor,
                            allostatic_load_increase: 0.03,
                            ..Default::default()
                        };
                        results.push((
                            effects,
                            Some(world.id),
                            CivEvent::new(
                                tick,
                                Some(world.id),
                                CivEventType::EmergencyDeclared,
                                format!(
                                    "{}: tidal flexing quake — structural stress ({:.1}% damage)",
                                    world.name,
                                    damage * flex_factor * 100.0
                                ),
                            ),
                        ));
                        self.total_disasters += 1;
                    }

                    // Ice shell instability (rare, catastrophic)
                    if rng.bernoulli(P_EUROPA_ICE_INSTABILITY) {
                        let severity = 0.1 + rng.next_f64() * 0.2; // 0.1-0.3
                        let effects = DisasterEffects {
                            infrastructure_damage: severity,
                            population_loss_fraction: severity * 0.15,
                            consciousness_shock: 0.2,
                            allostatic_load_increase: 0.3,
                            morale_impact: -0.4,
                            ..Default::default()
                        };
                        self.active_disasters.push(ActiveDisaster {
                            kind: DisasterKind::Planetary(
                                PlanetaryEventKind::EuropaIceShellInstability,
                            ),
                            severity,
                            remaining_ticks: 2,
                            world_id: Some(world.id),
                            effects: effects.clone(),
                        });
                        results.push((
                            effects,
                            Some(world.id),
                            CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                                format!("{}: ICE SHELL INSTABILITY — cryovolcanic event, severity {:.0}%",
                                    world.name, severity * 100.0)),
                        ));
                        self.total_disasters += 1;
                    }
                }
                "Titan" => {
                    // Heating failure — Titan's signature killer (2× thermal MTBF)
                    if rng.bernoulli(P_TITAN_HEATING_FAILURE) {
                        let (pop_loss, infra_damage) = if world.infrastructure_level < 0.5 {
                            // No redundant heating → freeze cascade
                            let severity = 0.1 + rng.next_f64() * 0.4; // 0.1-0.5
                            (severity, 0.1)
                        } else {
                            // Redundant heating absorbs the failure
                            (0.0, 0.05)
                        };
                        let effects = DisasterEffects {
                            population_loss_fraction: pop_loss,
                            infrastructure_damage: infra_damage,
                            allostatic_load_increase: 0.2,
                            morale_impact: -0.3,
                            consciousness_shock: pop_loss * 0.5,
                            ..Default::default()
                        };
                        results.push((
                            effects,
                            Some(world.id),
                            CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                                if pop_loss > 0.0 {
                                    format!("{}: HEATING SYSTEM FAILURE — freeze cascade, {:.0}% casualties",
                                        world.name, pop_loss * 100.0)
                                } else {
                                    format!("{}: heating system failure — redundant systems activated",
                                        world.name)
                                }),
                        ));
                        self.total_disasters += 1;
                    }

                    // Cryogenic embrittlement (cumulative)
                    // Cryogenic Materials tech: 70% reduction in embrittlement rate
                    let embrittlement_rate = if self.tech_tree.is_achieved("Cryogenic Materials") {
                        TITAN_EMBRITTLEMENT_PER_TICK * 0.3
                    } else {
                        TITAN_EMBRITTLEMENT_PER_TICK
                    };
                    self.titan_embrittlement =
                        (self.titan_embrittlement + embrittlement_rate).min(1.0);
                    // Tech level reduces embrittlement effects (better materials)
                    let embrittlement_factor = 1.0 - world.knowledge.mean_tech_level() * 0.3;
                    let seal_accel = 3.0 * embrittlement_factor.max(0.1);
                    if self.titan_embrittlement > 0.1 {
                        let effects = DisasterEffects {
                            infrastructure_damage: 0.005 * embrittlement_factor.max(0.1),
                            ..Default::default()
                        };
                        // Accelerate seal degradation for Titan
                        self.seal_degradation = (self.seal_degradation
                            + SEAL_DEGRADATION_PER_TICK * seal_accel)
                            .min(1.0);
                        if rng.bernoulli(0.1) {
                            // Log occasionally, not every tick
                            results.push((
                                effects,
                                Some(world.id),
                                CivEvent::new(
                                    tick,
                                    Some(world.id),
                                    CivEventType::EmergencyDeclared,
                                    format!(
                                        "{}: cryogenic embrittlement — seal degradation at {:.1}%",
                                        world.name,
                                        self.seal_degradation * 100.0
                                    ),
                                ),
                            ));
                        }
                    }

                    // Major methane rainstorm
                    if rng.bernoulli(P_TITAN_METHANE_STORM) {
                        let duration = 1 + (rng.next_f64() * 2.0) as u32; // 1-2 ticks
                        let effects = DisasterEffects {
                            resource_production_penalty: 0.2,
                            infrastructure_damage: 0.02,
                            allostatic_load_increase: 0.1,
                            morale_impact: -0.1,
                            ..Default::default()
                        };
                        self.active_disasters.push(ActiveDisaster {
                            kind: DisasterKind::Planetary(PlanetaryEventKind::TitanMethaneStorm),
                            severity: 0.4,
                            remaining_ticks: duration,
                            world_id: Some(world.id),
                            effects: effects.clone(),
                        });
                        results.push((
                            effects,
                            Some(world.id),
                            CivEvent::new(
                                tick,
                                Some(world.id),
                                CivEventType::EmergencyDeclared,
                                format!(
                                    "{}: major methane rainstorm — flooding, {} months duration",
                                    world.name, duration
                                ),
                            ),
                        ));
                        self.total_disasters += 1;
                    }

                    // Low-gravity chronic health degradation (deterministic, every tick)
                    let low_g_load = TITAN_LOW_G_LOAD_PER_TICK;
                    // Tech milestones reduce the health impact (centrifuge quarters, pharma)
                    let low_g_mitigation =
                        if self.tech_tree.is_achieved("Manufacturing Breakthrough") {
                            0.5
                        } else {
                            1.0
                        };
                    results.push((
                        DisasterEffects {
                            allostatic_load_increase: low_g_load * low_g_mitigation,
                            ..Default::default()
                        },
                        Some(world.id),
                        CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                            format!("{}: chronic 0.14g health effects — bone loss, cardiovascular stress",
                                world.name)),
                    ));

                    // Titan is IMMUNE to radiation disasters (double shielded)
                    // — no radiation events generated here; solar events in
                    //   roll_solar_events apply globally but Titan's atmosphere
                    //   (1085 g/cm²) and Saturn's magnetosphere absorb them.
                    //   This is handled in the disaster application phase.
                }
                "Earth" => {
                    // Mega-earthquake M9.0+ (~1 per 80 years)
                    if rng.bernoulli(P_MEGA_QUAKE) {
                        let severity = 0.05 + rng.next_f64() * 0.1; // 0.05-0.15
                        let mut effects = DisasterEffects {
                            infrastructure_damage: severity,
                            population_loss_fraction: 0.001 + rng.next_f64() * 0.009,
                            allostatic_load_increase: 0.15,
                            morale_impact: -0.2,
                            ..Default::default()
                        };
                        results.push((
                            effects.clone(),
                            Some(world.id),
                            CivEvent::new(
                                tick,
                                Some(world.id),
                                CivEventType::EmergencyDeclared,
                                format!(
                                    "{}: MEGA-QUAKE M9.0+ — infrastructure damage {:.1}%",
                                    world.name,
                                    severity * 100.0
                                ),
                            ),
                        ));
                        self.total_disasters += 1;

                        // 50% chance of mega-tsunami co-occurrence
                        if rng.bernoulli(0.5) {
                            effects = DisasterEffects {
                                infrastructure_damage: 0.05,
                                resource_production_penalty: 0.1,
                                population_loss_fraction: 0.002,
                                allostatic_load_increase: 0.1,
                                morale_impact: -0.15,
                                ..Default::default()
                            };
                            results.push((
                                effects,
                                Some(world.id),
                                CivEvent::new(
                                    tick,
                                    Some(world.id),
                                    CivEventType::EmergencyDeclared,
                                    format!("{}: MEGA-TSUNAMI triggered by quake", world.name),
                                ),
                            ));
                            self.total_disasters += 1;
                        }
                    }

                    // Supervolcanic eruption VEI 7+ (~1 per 80,000 years)
                    if rng.bernoulli(P_SUPERVOLCANO) {
                        let effects = DisasterEffects {
                            resource_production_penalty: 0.3,
                            allostatic_load_increase: 0.2,
                            consciousness_shock: 0.1,
                            morale_impact: -0.3,
                            ..Default::default()
                        };
                        // Volcanic winter: 24 ticks (2 years) of reduced food production
                        self.active_disasters.push(ActiveDisaster {
                            kind: DisasterKind::Planetary(
                                PlanetaryEventKind::EarthSupervolcanicEruption,
                            ),
                            severity: 0.9,
                            remaining_ticks: 24,
                            world_id: None, // Global: affects all worlds' solar/food
                            effects: effects.clone(),
                        });
                        results.push((
                            effects,
                            None,
                            CivEvent::new(tick, None, CivEventType::EmergencyDeclared,
                                "SUPERVOLCANIC ERUPTION VEI 7+ — volcanic winter begins (24 months)"),
                        ));
                        self.total_disasters += 1;
                    }

                    // Magnetosphere excursion effects (chronic, if active)
                    if self.magnetosphere.excursion_active {
                        results.push((
                            DisasterEffects {
                                allostatic_load_increase: 0.01,
                                resource_production_penalty: 0.05, // UV/ozone damage to agriculture
                                ..Default::default()
                            },
                            Some(world.id),
                            CivEvent::new(tick, Some(world.id), CivEventType::EmergencyDeclared,
                                format!("{}: magnetic field excursion — elevated surface radiation, ozone depletion",
                                    world.name)),
                        ));
                    }
                }
                _ => {} // Generic locations use only shared disaster categories
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

            // #4: Disaster interaction — existing damage makes MORE failures likely.
            // Low infrastructure = degraded systems = higher failure probability.
            // This creates compound cascades: quake → damage → ECLSS failure → crisis.
            let active_count = self.active_per_world.get(&world.id).copied().unwrap_or(0);
            let interaction_amplifier = if active_count >= 2 {
                1.0 + (active_count as f64 - 1.0) * 0.3 // Each active disaster +30% failure risk
            } else {
                1.0
            };
            // Low infrastructure = degraded systems
            let damage_amplifier = if world.infrastructure_level < 0.5 {
                1.0 + (0.5 - world.infrastructure_level) * 2.0 // Up to 2x at infra=0
            } else {
                1.0
            };

            // Mechanism 1 — Tech-to-MTBF (Infrastructure Shield): higher technology
            // extends mean time between failures, and mature colonies (infra > 0.7)
            // get an additional 50% MTBF bonus from built-in redundancy.
            let tech_multiplier = 1.0 + world.knowledge.mean_tech_level() * 2.0;
            let infra_redundancy = if world.infrastructure_level > 0.7 {
                1.5
            } else {
                1.0
            };
            let effective_mtbf_scale = tech_multiplier * infra_redundancy;
            // Scale failure probability by inverse of infrastructure level
            // Better infrastructure = more redundancy (original factor)
            let infra_factor = 1.0 / (0.5 + world.infrastructure_level);
            // Combined: divide base probability by MTBF scale, amplified by disaster interactions
            let mtbf_factor =
                (infra_factor / effective_mtbf_scale) * interaction_amplifier * damage_amplifier;

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
                if rng.bernoulli(prob * mtbf_factor) {
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
                    CivEvent::new(
                        tick,
                        Some(world.id),
                        CivEventType::FactionConflict,
                        format!("{}: interpersonal conflict — productivity loss", world.name),
                    ),
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
                    CivEvent::new(
                        tick,
                        Some(world.id),
                        CivEventType::EmergencyDeclared,
                        format!(
                            "{}: cognitive impairment episode — decision quality degraded",
                            world.name
                        ),
                    ),
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
                    CivEvent::new(
                        tick,
                        Some(world.id),
                        CivEventType::FactionEmerged,
                        format!(
                            "{}: social cohesion collapse — faction emergence likely",
                            world.name
                        ),
                    ),
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
                    CivEvent::new(
                        tick,
                        Some(world.id),
                        CivEventType::EmergencyDeclared,
                        format!(
                            "{}: individual psychotic break — medical emergency",
                            world.name
                        ),
                    ),
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
                    CivEvent::new(
                        tick,
                        Some(world.id),
                        CivEventType::GovernanceTransition,
                        format!("{}: authority challenge — leadership crisis", world.name),
                    ),
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
                CivEvent::new(
                    tick,
                    None,
                    CivEventType::InnovationStagnation,
                    "Technology regression: researchers below critical mass — knowledge loss",
                ),
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
                    CivEvent::new(
                        tick,
                        None,
                        CivEventType::InnovationBreakthrough,
                        format!("TECH MILESTONE: {} achieved", milestone.name),
                    ),
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
                        !(ev.tick == tick && ev.description.contains(&milestone.name))
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
                        CivEvent::new(
                            tick,
                            Some(world.id),
                            CivEventType::ResourceCrisis,
                            format!("{}: resource depletion crisis compounding", world.name),
                        ),
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
                        CivEvent::new(
                            tick,
                            Some(world.id),
                            CivEventType::ConstitutionalCalcification,
                            format!(
                                "{}: institutional sclerosis — governance rigidity at {:.0}%",
                                world.name,
                                calcification * 100.0
                            ),
                        ),
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
                    CivEvent::new(
                        tick,
                        Some(world.id),
                        CivEventType::EmergencyDeclared,
                        format!(
                            "{}: SYSTEMIC CASCADE FAILURE — {} infrastructure failures in {} ticks",
                            world.name, self.cascade_failure_count, CASCADE_WINDOW_TICKS
                        ),
                    ),
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
            if let Some(entry) = self
                .high_load_ticks
                .iter_mut()
                .find(|(id, _)| *id == world.id)
            {
                if mean_load > COHESION_CRISIS_LOAD {
                    entry.1 += 1;
                } else {
                    entry.1 = 0;
                }
            } else {
                self.high_load_ticks.push((
                    world.id,
                    if mean_load > COHESION_CRISIS_LOAD {
                        1
                    } else {
                        0
                    },
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
                if let Some(entry) = self
                    .high_load_ticks
                    .iter_mut()
                    .find(|(id, _)| *id == world.id)
                {
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
        // Mechanism 5 — Collective Memory Inoculation: record disaster kinds
        // that have fully resolved (remaining_ticks == 0) so that subsequent
        // occurrences of the same kind receive a 30% severity reduction.
        for d in &self.active_disasters {
            if d.remaining_ticks == 0 {
                self.survived_disaster_types
                    .insert(Self::disaster_kind_key(d.kind));
            }
        }
        self.active_disasters.retain(|d| d.remaining_ticks > 0);
    }

    /// Stable string key for a disaster kind, used for collective memory inoculation.
    fn disaster_kind_key(kind: DisasterKind) -> String {
        format!("{:?}", kind)
    }

    /// Apply collective memory inoculation: reduce severity by 30% if the
    /// civilization has survived this disaster kind before (Mechanism 5).
    fn inoculation_factor(&self, kind: DisasterKind) -> f64 {
        if self
            .survived_disaster_types
            .contains(&Self::disaster_kind_key(kind))
        {
            0.7
        } else {
            1.0
        }
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
        // O2 generator MTBF from empirical data → p ≈ 1/MTBF
        let expected_o2 = 1.0 / crate::empirical::ECLSS_O2_GEN_MTBF_MONTHS;
        assert!(
            (P_O2_FAILURE - expected_o2).abs() < 0.001,
            "O2 failure prob {:.4} should match 1/{} = {:.4}",
            P_O2_FAILURE,
            crate::empirical::ECLSS_O2_GEN_MTBF_MONTHS,
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
        assert!(
            triggered,
            "Tainter diminishing returns should trigger eventually"
        );
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
        assert_eq!(tree.milestones.len(), 16); // 6 original + 10 extended (incl. Genetic Engineering)
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

    #[test]
    fn test_day_side_solar_does_more_damage() {
        // Fix 10: Continuous Time Phase
        let mut day_effects = DisasterEffects {
            population_loss_fraction: 0.1,
            infrastructure_damage: 0.2,
            resource_production_penalty: 0.1,
            solar_power_penalty: 0.5,
            consciousness_shock: 0.1,
            allostatic_load_increase: 0.0,
            electronics_damage: 0.3,
            morale_impact: -0.2,
        };
        let mut night_effects = day_effects.clone();

        // Day-side (phase 0.2): full damage
        day_effects.apply_event_phase(0.2, true);
        // Night-side (phase 0.7): reduced solar damage
        night_effects.apply_event_phase(0.7, true);

        assert!(
            day_effects.electronics_damage > night_effects.electronics_damage,
            "Day-side solar should do more electronics damage: {} vs {}",
            day_effects.electronics_damage,
            night_effects.electronics_damage
        );
        assert!(
            day_effects.population_loss_fraction > night_effects.population_loss_fraction,
            "Day-side solar should cause more casualties: {} vs {}",
            day_effects.population_loss_fraction,
            night_effects.population_loss_fraction
        );
    }

    #[test]
    fn test_night_shift_slower_response() {
        // Fix 10: Night shift increases infrastructure damage
        let mut effects = DisasterEffects {
            population_loss_fraction: 0.0,
            infrastructure_damage: 0.2,
            resource_production_penalty: 0.1,
            solar_power_penalty: 0.0,
            consciousness_shock: 0.0,
            allostatic_load_increase: 0.0,
            electronics_damage: 0.0,
            morale_impact: 0.0,
        };

        let base_infra = effects.infrastructure_damage;
        effects.apply_event_phase(0.8, false); // Night shift, non-solar

        assert!(
            effects.infrastructure_damage > base_infra,
            "Night shift should amplify infra damage: {} vs {}",
            effects.infrastructure_damage,
            base_infra
        );
    }
}
