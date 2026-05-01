// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spaceport Funnel: translates aggregate Earth statistics into individual
//! FEP/HDC agents for off-world colonization.
//!
//! This module implements the mathematical bridge between the macro-level
//! regional demographics model (`earth_regions.rs`) and the micro-level
//! individual agent simulation. When colonists depart Earth, the Spaceport
//! Funnel:
//!
//! 1. Samples Phi from the source region's distribution: N(phi_mean, phi_sd)
//! 2. Samples skills biased by the region's economic specialization
//! 3. Samples education from the region's education index
//! 4. Deducts launch cost from the region's GDP
//! 5. Applies brain drain: losing above-average individuals lowers regional Phi
//!
//! Kessler Syndrome (orbital debris cascade) can block launches entirely,
//! severing the supply line from Earth to off-world colonies.

use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
use crate::earth_regions::EarthRegion;
use crate::needs::PsychologicalNeeds;
use crate::stochastic::StochasticEngine;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default launch cost per kg to LEO in USD (2025 SpaceX Starship target).
const DEFAULT_LAUNCH_COST_PER_KG: f64 = 100.0;

/// Average colonist mass in kg (including life support equipment).
const COLONIST_MASS_KG: f64 = 200.0;

/// Default throughput: colonists per year for a mature equatorial spaceport.
const DEFAULT_THROUGHPUT_PER_YEAR: usize = 1000;

/// Equatorial delta-v bonus in km/s (Earth rotation at equator).
const EQUATORIAL_DV_BONUS: f64 = 0.46;

/// Construction time in ticks for a new spaceport.
const SPACEPORT_CONSTRUCTION_TICKS: u32 = 60; // 5 years

/// Default Kessler syndrome duration range in ticks.
const KESSLER_MIN_DURATION: u32 = 60; // 5 years
const KESSLER_MAX_DURATION: u32 = 120; // 10 years

/// Off-world resource production penalty when supply line is cut.
const SUPPLY_LINE_CUT_PENALTY: f64 = 0.30;

/// Colonists are self-selected: sample from above the mean by this many SDs.
const COLONIST_PHI_SELECTION_BIAS: f64 = 1.5;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A spaceport facility for launching colonists to off-world settlements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spaceport {
    /// Region this spaceport is located in (e.g., "Latin America" for Ecuador).
    pub location_region: String,
    /// Delta-v bonus from latitude (km/s). 0.46 for equatorial.
    pub delta_v_bonus_km_s: f64,
    /// Maximum colonists that can be launched per year.
    pub throughput_per_year: usize,
    /// Launch cost per kg to LEO in USD.
    pub launch_cost_per_kg: f64,
    /// Whether the spaceport is operational.
    pub operational: bool,
    /// Tick at which construction began.
    pub construction_start_tick: u32,
    /// Tick at which construction completed (None if still building).
    pub construction_complete_tick: Option<u32>,
    /// Kessler syndrome: if true, launches are blocked.
    pub leo_blocked: bool,
    /// Remaining ticks of Kessler blockade.
    pub kessler_duration_remaining: u32,
    /// Cumulative colonists launched.
    pub total_launched: u64,
}

/// Description of a colonist selection batch for launch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColonistSelection {
    /// Source region name.
    pub source_region: String,
    /// Number of colonists to instantiate.
    pub count: usize,
    /// Mean Phi of the selected batch.
    pub mean_phi: f64,
    /// Mean education level of the batch.
    pub mean_education: f64,
    /// Skill bias from the source region's specialization.
    pub skill_bias: [f64; 8],
    /// Economic impact on the source region (GDP loss from brain drain).
    pub brain_drain_cost: f64,
}

impl Spaceport {
    /// Create the default equatorial spaceport (e.g., Ecuador/Kourou).
    pub fn new_equatorial(start_tick: u32) -> Self {
        Self {
            location_region: "Latin America".into(),
            delta_v_bonus_km_s: EQUATORIAL_DV_BONUS,
            throughput_per_year: DEFAULT_THROUGHPUT_PER_YEAR,
            launch_cost_per_kg: DEFAULT_LAUNCH_COST_PER_KG,
            operational: false,
            construction_start_tick: start_tick,
            construction_complete_tick: None,
            leo_blocked: false,
            kessler_duration_remaining: 0,
            total_launched: 0,
        }
    }

    /// Tick the spaceport state (construction progress, Kessler countdown).
    pub fn tick(&mut self, current_tick: u32) {
        // Construction progress.
        if !self.operational && self.construction_complete_tick.is_none() {
            if current_tick >= self.construction_start_tick + SPACEPORT_CONSTRUCTION_TICKS {
                self.construction_complete_tick = Some(current_tick);
                self.operational = true;
            }
        }

        // Kessler countdown.
        if self.leo_blocked {
            if self.kessler_duration_remaining > 0 {
                self.kessler_duration_remaining -= 1;
            }
            if self.kessler_duration_remaining == 0 {
                self.leo_blocked = false;
            }
        }
    }

    /// Whether the spaceport can currently launch.
    pub fn can_launch(&self) -> bool {
        self.operational && !self.leo_blocked
    }

    /// Activate Kessler syndrome blockade.
    pub fn activate_kessler(&mut self, duration_ticks: u32) {
        self.leo_blocked = true;
        self.kessler_duration_remaining = duration_ticks;
    }

    /// Maximum colonists this spaceport can launch this tick (monthly slice).
    pub fn monthly_capacity(&self) -> usize {
        if !self.can_launch() {
            return 0;
        }
        // throughput_per_year / 12, minimum 1 if operational
        (self.throughput_per_year / 12).max(1)
    }
}

// ---------------------------------------------------------------------------
// Spaceport Funnel: macro → micro translation
// ---------------------------------------------------------------------------

/// Instantiate individual colonist agents from aggregate regional data.
///
/// This is the Spaceport Funnel integral — the mathematical bridge from
/// Earth's aggregate demographics to individual FEP/HDC agents.
///
/// For each colonist:
/// 1. Sample Phi from source region's distribution: N(phi_mean, phi_sd)
/// 2. Sample skills biased by region's economic specialization
/// 3. Sample education from region's education_index
/// 4. Apply launch cost: deduct from region's GDP
/// 5. Brain drain: reduce source region's phi_mean slightly
pub fn instantiate_colonists(
    selection: &ColonistSelection,
    source_region: &mut EarthRegion,
    destination_world_id: u32,
    current_tick: u32,
    next_agent_id: &mut u64,
    rng: &mut StochasticEngine,
) -> Vec<CivAgent> {
    let mut agents = Vec::with_capacity(selection.count);

    for _ in 0..selection.count {
        // 1. Sample Phi from the source region's distribution.
        //    Colonists are self-selected (above-average Phi).
        let phi = rng
            .next_gaussian(
                source_region.phi_distribution_mean
                    + COLONIST_PHI_SELECTION_BIAS * source_region.phi_distribution_sd,
                source_region.phi_distribution_sd,
            )
            .clamp(0.05, 0.95);

        // 2. Sample skills biased by the region's economic specialization.
        let mut skills = SkillVector::new();
        let mut skill_vals = [0.0f64; 8];
        for (i, &spec) in source_region.economic_specialization.iter().enumerate() {
            // Base skill + regional bias + random noise
            let base = 0.1 + spec * 0.5;
            skill_vals[i] = rng.next_gaussian(base, 0.1).clamp(0.05, 1.0);
        }
        // Apply to SkillVector
        skills.engineering = skill_vals[0];
        skills.agriculture = skill_vals[1];
        skills.medicine = skill_vals[2];
        skills.governance = skill_vals[3];
        skills.science = skill_vals[4];
        skills.education = skill_vals[5];
        skills.art_culture = skill_vals[6];
        skills.logistics = skill_vals[7];

        // 3. Sample education from the region's education_index.
        let education = rng
            .next_gaussian(source_region.education_index, 0.1)
            .clamp(0.05, 1.0);

        // 4. Build consciousness state from sampled Phi.
        let consciousness = ConsciousnessState {
            level: (phi * 1.2).clamp(0.0, 1.0),
            meta_awareness: (phi * 0.8).clamp(0.0, 1.0),
            coherence: (phi * 0.9).clamp(0.0, 1.0),
            care_activation: rng.next_gaussian(0.4, 0.15).clamp(0.0, 1.0),
            harmonic_alignment: (phi * 0.7).clamp(0.0, 1.0),
            epistemic_confidence: (education * 0.8).clamp(0.0, 1.0),
        };

        // 5. Assign sex.
        let sex = if rng.bernoulli(0.5) {
            BiologicalSex::Male
        } else {
            BiologicalSex::Female
        };

        // 6. Age: colonists are adults (25-45 years old).
        let age_months = (25 * 12) + (rng.next_u64() % (20 * 12)) as u32;
        let birth_tick = current_tick.wrapping_sub(age_months);

        let agent = CivAgent {
            id: *next_agent_id,
            birth_tick,
            death_tick: None,
            sex,
            world_id: destination_world_id,
            health: rng.next_gaussian(0.90, 0.08).clamp(0.4, 1.0),
            skills,
            education_level: education,
            consciousness,
            partner_id: None,
            children_ids: vec![],
            is_immigrant: true,
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
            ethics: crate::agent::EthicalOrientation::from_culture(
                source_region.cultural_profile.individualism,
                rng,
            ),
            // 8D sovereign profile sampled from colonist's source-region culture.
            sovereign_profile: crate::sovereign_profile::SovereignProfile::sample(
                source_region.cultural_profile.individualism,
                rng,
            ),
            justice: crate::sub_passport::RestorativeJustice::new(),
        };
        *next_agent_id += 1;
        agents.push(agent);
    }

    // Apply brain drain to source region.
    apply_brain_drain(source_region, selection.count);

    // Deduct launch cost from region GDP.
    let total_launch_cost = selection.count as f64
        * COLONIST_MASS_KG
        * source_region.gdp_per_capita.min(DEFAULT_LAUNCH_COST_PER_KG);
    // Express as per-capita GDP reduction (spread across the whole population).
    let gdp_deduction = total_launch_cost / (source_region.population * 1_000_000.0);
    source_region.gdp_per_capita = (source_region.gdp_per_capita - gdp_deduction).max(100.0);

    agents
}

/// Apply brain drain to a source region when colonists depart.
///
/// Colonists are above-average individuals (self-selected). Losing them
/// reduces the regional average Phi.
///
/// Formula:
/// ```text
/// phi_departing = region.phi_mean + 1.5 * region.phi_sd
/// brain_drain = (phi_departing - region.phi_mean) * (count / population_individuals)
/// region.phi_mean -= brain_drain
/// ```
pub fn apply_brain_drain(region: &mut EarthRegion, count: usize) {
    if region.population <= 0.0 || count == 0 {
        return;
    }
    let phi_of_departing =
        region.phi_distribution_mean + COLONIST_PHI_SELECTION_BIAS * region.phi_distribution_sd;
    let brain_drain = (phi_of_departing - region.phi_distribution_mean)
        * (count as f64 / (region.population * 1_000_000.0));
    region.phi_distribution_mean = (region.phi_distribution_mean - brain_drain).max(0.05);

    // Also reduce population.
    let pop_loss = count as f64 / 1_000_000.0;
    region.population = (region.population - pop_loss).max(0.1);
}

/// Prepare a colonist selection from a specific region.
///
/// Returns None if the region cannot supply colonists (too small, no access).
pub fn prepare_selection(region: &EarthRegion, count: usize) -> Option<ColonistSelection> {
    // Minimum population to avoid depopulation: 1M
    if region.population < 1.0 {
        return None;
    }
    // Can't launch more than 0.01% of the population per batch
    let max_batch = (region.population * 1_000_000.0 * 0.0001) as usize;
    let actual_count = count.min(max_batch).max(1);

    let phi_of_departing =
        region.phi_distribution_mean + COLONIST_PHI_SELECTION_BIAS * region.phi_distribution_sd;
    let brain_drain_cost = (phi_of_departing - region.phi_distribution_mean)
        * (actual_count as f64 / (region.population * 1_000_000.0));

    Some(ColonistSelection {
        source_region: region.name.clone(),
        count: actual_count,
        mean_phi: phi_of_departing,
        mean_education: region.education_index,
        skill_bias: region.economic_specialization,
        brain_drain_cost,
    })
}

/// Apply supply line cut penalty to an off-world colony's resource production.
///
/// When Kessler syndrome blocks LEO, off-world colonies lose their supply
/// line and resource production drops by 30%.
pub fn apply_supply_line_cut_penalty() -> f64 {
    SUPPLY_LINE_CUT_PENALTY
}

/// Determine Kessler duration in ticks given RNG.
pub fn kessler_duration(rng: &mut StochasticEngine) -> u32 {
    let range = KESSLER_MAX_DURATION - KESSLER_MIN_DURATION;
    KESSLER_MIN_DURATION + (rng.next_u64() % range as u64) as u32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::earth_regions::build_earth_regions;

    #[test]
    fn test_spaceport_construction_and_operation() {
        let mut sp = Spaceport::new_equatorial(0);
        assert!(!sp.operational);
        assert!(!sp.can_launch());

        // Tick through construction period
        for tick in 0..SPACEPORT_CONSTRUCTION_TICKS {
            sp.tick(tick);
        }
        sp.tick(SPACEPORT_CONSTRUCTION_TICKS);
        assert!(sp.operational);
        assert!(sp.can_launch());
    }

    #[test]
    fn test_kessler_blocks_launches() {
        let mut sp = Spaceport::new_equatorial(0);
        // Make operational
        sp.operational = true;
        sp.construction_complete_tick = Some(0);
        assert!(sp.can_launch());

        // Activate Kessler
        sp.activate_kessler(10);
        assert!(!sp.can_launch());
        assert!(sp.leo_blocked);

        // Tick down
        for tick in 1..=10 {
            sp.tick(tick);
        }
        assert!(!sp.leo_blocked);
        assert!(sp.can_launch());
    }

    #[test]
    fn test_instantiate_colonists_correct_phi_distribution() {
        let mut regions = build_earth_regions();
        let mut rng = StochasticEngine::new(42);
        let na = regions
            .iter()
            .position(|r| r.name == "North America")
            .unwrap();

        let selection = prepare_selection(&regions[na], 50).unwrap();
        assert_eq!(selection.count, 50);

        let mut next_id = 0u64;
        let agents =
            instantiate_colonists(&selection, &mut regions[na], 1, 0, &mut next_id, &mut rng);

        assert_eq!(agents.len(), 50);
        // Mean Phi of colonists should be above the region mean (self-selected).
        let mean_phi: f64 =
            agents.iter().map(|a| a.consciousness.phi()).sum::<f64>() / agents.len() as f64;
        assert!(
            mean_phi > regions[na].phi_distribution_mean,
            "Colonist mean Phi {mean_phi} should exceed region mean {}",
            regions[na].phi_distribution_mean
        );
    }

    #[test]
    fn test_brain_drain_reduces_source_phi() {
        let mut regions = build_earth_regions();
        let na = regions
            .iter()
            .position(|r| r.name == "North America")
            .unwrap();
        let initial_phi = regions[na].phi_distribution_mean;

        apply_brain_drain(&mut regions[na], 10_000);

        assert!(
            regions[na].phi_distribution_mean < initial_phi,
            "Brain drain should reduce source region Phi"
        );
    }

    #[test]
    fn test_supply_line_cut_penalty() {
        let penalty = apply_supply_line_cut_penalty();
        assert!((penalty - 0.30).abs() < 0.001);
    }

    #[test]
    fn test_monthly_capacity_zero_when_blocked() {
        let mut sp = Spaceport::new_equatorial(0);
        sp.operational = true;
        sp.construction_complete_tick = Some(0);
        sp.activate_kessler(10);
        assert_eq!(sp.monthly_capacity(), 0);
    }

    #[test]
    fn test_kessler_duration_in_range() {
        let mut rng = StochasticEngine::new(42);
        for _ in 0..100 {
            let d = kessler_duration(&mut rng);
            assert!(d >= KESSLER_MIN_DURATION && d < KESSLER_MAX_DURATION + 1);
        }
    }
}
