// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 8-sector Cobb-Douglas production model with demurrage currency.
//!
//! Sectors match [`SkillVector`] indices: 0=engineering, 1=agriculture, 2=medicine,
//! 3=governance, 4=science, 5=education, 6=art_culture, 7=logistics.
//!
//! Production follows a Cobb-Douglas function with labor-intensive exponents
//! (alpha=0.7) appropriate for early space colonies where labor is the scarce factor.
//! Demurrage discourages hoarding and encourages reinvestment.

use serde::{Deserialize, Serialize};

use crate::agent::{CivAgent, SKILL_SECTORS};
use crate::viability::ScalingFactors;
use crate::world::WorldResources;

/// Number of economic sectors (matches [`SKILL_SECTORS`]).
pub const NUM_SECTORS: usize = 8;

/// Labor share exponent in Cobb-Douglas production (alpha).
const LABOR_EXPONENT: f64 = 0.7;

/// Capital share exponent in Cobb-Douglas production.
/// When energy is included: α=0.60 labor, β=0.25 capital, γ=0.15 energy (sum=1.0).
/// When energy is NOT included (legacy): α=0.7, β=0.3.
const CAPITAL_EXPONENT: f64 = 0.3;

/// Energy share exponent in extended Cobb-Douglas production.
/// Output = A × L^α × K^β × E^γ where γ captures energy's role in production.
/// CITATION: Stern (2011) "The role of energy in economic growth",
/// Annals of the New York Academy of Sciences 1219(1), pp. 26-51.
/// LIMITATION: Exponent is approximate. Real economies show γ ≈ 0.10-0.20.
const ENERGY_EXPONENT: f64 = 0.15;

/// When energy factor is active, labor and capital exponents are rescaled
/// to maintain constant returns to scale (α + β + γ = 1.0).
const LABOR_EXPONENT_WITH_ENERGY: f64 = 0.60;
const CAPITAL_EXPONENT_WITH_ENERGY: f64 = 0.25;

/// Minimum workforce fraction per sector when population > 20.
const MIN_SECTOR_FRACTION: f64 = 0.05;

/// Default annual demurrage rate (2%).
const DEFAULT_DEMURRAGE_RATE: f64 = 0.02;

fn default_prices() -> [f64; NUM_SECTORS] {
    [1.0; NUM_SECTORS]
}

/// 8-sector economy with Cobb-Douglas production and demurrage currency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldEconomy {
    /// Units produced per tick per sector.
    pub sector_output: [f64; NUM_SECTORS],
    /// Agents assigned per sector.
    pub sector_workers: [usize; NUM_SECTORS],
    /// Output per worker (grows with technology).
    pub labor_productivity: [f64; NUM_SECTORS],
    /// Technology multiplier from knowledge system (A in Cobb-Douglas).
    pub technology_multiplier: [f64; NUM_SECTORS],
    /// Infrastructure capital (K in Cobb-Douglas), range 0.0-100.0.
    pub infrastructure_capital: f64,
    /// Total currency in circulation.
    pub currency_supply: f64,
    /// Annual demurrage rate (default 0.02 = 2%).
    pub demurrage_rate: f64,
    /// Gini coefficient measuring inequality (0.0-1.0).
    pub gini_coefficient: f64,
    /// Sum of all sector outputs this tick.
    pub total_production: f64,
    /// Local production / total consumption ratio.
    pub self_sufficiency: f64,
    /// Net exports (positive = trade surplus).
    pub trade_balance: f64,
    /// Fraction of output reinvested in infrastructure (0.0-0.5).
    pub investment_rate: f64,
    /// Fix 3: Price per sector unit (supply/demand driven).
    #[serde(default = "default_prices")]
    pub prices: [f64; NUM_SECTORS],
    /// Fix 3: Inflation rate (weighted mean price change per tick).
    #[serde(default)]
    pub inflation_rate: f64,
    /// Sectors currently in skill gap crisis (zero workers AND pop > 10).
    /// Requires 36 ticks of education to recover.
    #[serde(default)]
    pub skill_gap_sectors: Vec<usize>,
    /// Per-sector: ticks remaining until skill gap recovery (0 = no gap or recovered).
    #[serde(default)]
    pub skill_gap_recovery_ticks: [u32; NUM_SECTORS],
}

impl WorldEconomy {
    /// Create a new economy at subsistence level.
    pub fn new() -> Self {
        Self {
            sector_output: [0.0; NUM_SECTORS],
            sector_workers: [0; NUM_SECTORS],
            labor_productivity: [1.0; NUM_SECTORS],
            technology_multiplier: [1.0; NUM_SECTORS],
            infrastructure_capital: 1.0,
            currency_supply: 1000.0,
            demurrage_rate: DEFAULT_DEMURRAGE_RATE,
            gini_coefficient: 0.0,
            total_production: 0.0,
            self_sufficiency: 0.0,
            trade_balance: 0.0,
            investment_rate: 0.15,
            prices: [1.0; NUM_SECTORS],
            inflation_rate: 0.0,
            skill_gap_sectors: Vec::new(),
            skill_gap_recovery_ticks: [0; NUM_SECTORS],
        }
    }

    /// Assign workers to sectors based on their strongest skill.
    ///
    /// Each working-age agent is placed in the sector matching their highest skill.
    /// If population > 20, a minimum of 5% of workers is guaranteed per sector
    /// by reassigning excess workers from over-represented sectors.
    pub fn assign_workers(&mut self, agents: &[CivAgent], current_tick: u32) {
        self.sector_workers = [0; NUM_SECTORS];

        // Count workers by strongest skill, with coordination-aware reallocation.
        // Agents with high coordination_understanding (systems thinking) detect
        // which sectors are understaffed and probabilistically reallocate there,
        // even if it's not their strongest skill. This models the capacity to
        // see the system as a whole rather than just optimizing individually.
        let mut worker_count = 0usize;
        // First pass: count natural assignments to detect bottlenecks.
        // Ethics-aware: sector attractiveness = skill + ethical affinity.
        // Virtue/care agents gravitate toward medicine/education even if their
        // raw skill is slightly weaker there (Gilligan 1982; vocation as calling).
        let mut natural_assignment = Vec::new();
        for agent in agents.iter().filter(|a| a.is_alive()) {
            let stage = agent.life_stage(current_tick);
            if !stage.can_work() {
                continue;
            }
            worker_count += 1;
            let skills = agent.skills.as_slice();
            let best_sector = skills
                .iter()
                .enumerate()
                .map(|(i, &s)| (i, s + agent.ethics.sector_affinity(i)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            natural_assignment.push((best_sector, agent.coordination_understanding));
            self.sector_workers[best_sector] += 1;
        }

        // Second pass: coordination-aware reallocation.
        // Agents with cu > 0.3 identify the most understaffed sector and have
        // a cu-proportional probability of switching to it.
        if worker_count > 20 {
            let min_sector = (0..NUM_SECTORS)
                .min_by_key(|&i| self.sector_workers[i])
                .unwrap_or(0);
            let min_workers = self.sector_workers[min_sector];
            let avg_workers = worker_count / NUM_SECTORS;

            // Only reallocate if there's a genuine bottleneck (< 50% of average)
            if min_workers < avg_workers / 2 {
                for &(natural, cu) in &natural_assignment {
                    if cu > 0.3 && natural != min_sector {
                        // Probability of switching = cu * 0.15 (max 7.5% chance)
                        // This is deliberately modest — systems thinking helps
                        // but doesn't override personal specialization entirely.
                        let switch_prob = cu * 0.15;
                        // Deterministic approximation: switch if cu high enough
                        // and this sector is overstaffed (> 150% of average)
                        if self.sector_workers[natural] > avg_workers * 3 / 2 && switch_prob > 0.05
                        {
                            self.sector_workers[natural] -= 1;
                            self.sector_workers[min_sector] += 1;
                        }
                    }
                }
            }
        }

        // Enforce minimum 5% per sector for populations > 20
        if worker_count > 20 {
            let min_workers = ((worker_count as f64 * MIN_SECTOR_FRACTION).ceil() as usize).max(1);

            // Iteratively redistribute: take from over-represented, give to under-represented
            for _pass in 0..NUM_SECTORS {
                let deficit_sectors: Vec<usize> = (0..NUM_SECTORS)
                    .filter(|&i| self.sector_workers[i] < min_workers)
                    .collect();

                if deficit_sectors.is_empty() {
                    break;
                }

                for &deficit_idx in &deficit_sectors {
                    while self.sector_workers[deficit_idx] < min_workers {
                        // Find the most over-represented sector
                        if let Some(donor) = (0..NUM_SECTORS)
                            .filter(|&i| self.sector_workers[i] > min_workers)
                            .max_by_key(|&i| self.sector_workers[i])
                        {
                            self.sector_workers[donor] -= 1;
                            self.sector_workers[deficit_idx] += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Run one tick of Cobb-Douglas production.
    ///
    /// **Legacy (no energy/scaling)**:
    /// `output_i = tech_mult_i * (effective_labor_i)^0.7 * infrastructure^0.3`
    ///
    /// **Extended (with energy + West-Bettencourt scaling)**:
    /// `output_i = tech_mult_i * scaling.innovation * (effective_labor_i)^0.60 * infrastructure^0.25 * energy^0.15`
    ///
    /// The extended form adds energy as a production factor (Stern 2011) and applies
    /// superlinear scaling from West-Bettencourt (2007): larger populations produce
    /// MORE per capita due to network effects and knowledge spillovers.
    pub fn tick_production(&mut self) {
        self.tick_production_extended(None, None);
    }

    /// Extended production with optional scaling and energy factors.
    ///
    /// - `scaling`: West-Bettencourt scaling factors (superlinear innovation boost).
    ///   When None, falls back to legacy exponents (α=0.7, β=0.3).
    /// - `energy_available`: Net energy available for production (from ViabilityEngine).
    ///   When None, energy is not a production constraint (legacy behavior).
    pub fn tick_production_extended(
        &mut self,
        scaling: Option<&ScalingFactors>,
        energy_available: Option<f64>,
    ) {
        self.total_production = 0.0;

        // Choose exponents based on whether energy is modeled
        let (alpha, beta, gamma) = if energy_available.is_some() {
            (
                LABOR_EXPONENT_WITH_ENERGY,
                CAPITAL_EXPONENT_WITH_ENERGY,
                ENERGY_EXPONENT,
            )
        } else {
            (LABOR_EXPONENT, CAPITAL_EXPONENT, 0.0)
        };

        // Innovation multiplier from scaling laws (default 1.0 if not provided)
        let innovation_mult = scaling.map_or(1.0, |s| s.innovation_multiplier);

        // Energy factor: normalized to [0, ∞) where 1.0 = baseline adequate
        // Energy of 0 should heavily suppress output but not zero it entirely
        // (people can still do manual labor without electricity)
        let energy_factor = match energy_available {
            Some(e) if gamma > 0.0 => {
                // Normalize: 100 ARU = baseline adequate for ~100 people
                let normalized = (e / 100.0).max(0.01);
                normalized.powf(gamma)
            }
            _ => 1.0,
        };

        for i in 0..NUM_SECTORS {
            let workers = self.sector_workers[i] as f64;
            if workers == 0.0 {
                self.sector_output[i] = 0.0;
                continue;
            }

            let effective_labor = workers * self.labor_productivity[i];
            let output = self.technology_multiplier[i]
                * innovation_mult
                * effective_labor.powf(alpha)
                * self.infrastructure_capital.max(0.01).powf(beta)
                * energy_factor;

            self.sector_output[i] = output;
            self.total_production += output;
        }
    }

    /// Apply ethics-based resource efficiency modifier to total production.
    /// Called after tick_production with the world's mean ethical orientation.
    /// Relational societies share resources more effectively (Ubuntu surplus flows).
    /// Virtue/care societies reduce waste through stewardship.
    /// Consequentialist societies extract efficiently but create waste externalities.
    /// Deontological societies add bureaucratic overhead but ensure fair distribution.
    pub fn apply_ethics_efficiency(&mut self, mean_ethics: &crate::agent::EthicalOrientation) {
        let modifier = 1.0 + mean_ethics.relational * 0.05 + mean_ethics.virtue_care * 0.03
            - mean_ethics.consequentialist * 0.025
            - mean_ethics.deontological * 0.02;
        let modifier = modifier.clamp(0.85, 1.15);
        self.total_production *= modifier;
        for output in &mut self.sector_output {
            *output *= modifier;
        }
    }

    /// Fix 3: Update prices based on supply/demand imbalance.
    ///
    /// `price[s] = 1.0 + (demand[s] - supply[s]) / supply[s].max(1.0)`
    ///
    /// Demand is proportional to worker count in the sector (consumers need
    /// what they don't produce). Supply is sector_output.
    pub fn tick_prices(&mut self, total_workers: usize) {
        let old_prices = self.prices;
        let demand_per_sector = (total_workers as f64) / NUM_SECTORS as f64;

        for i in 0..NUM_SECTORS {
            let supply = self.sector_output[i].max(1.0);
            let demand = demand_per_sector; // simplified: equal demand across sectors
            self.prices[i] = (1.0 + (demand - supply) / supply).clamp(0.1, 10.0);
        }

        // Inflation = mean price change
        let price_change: f64 = (0..NUM_SECTORS)
            .map(|i| self.prices[i] - old_prices[i])
            .sum::<f64>()
            / NUM_SECTORS as f64;
        self.inflation_rate = price_change;
    }

    /// Apply monthly demurrage to the currency supply.
    ///
    /// `currency_supply *= (1.0 - demurrage_rate / 12.0)`
    pub fn tick_demurrage(&mut self) {
        self.currency_supply *= 1.0 - self.demurrage_rate / 12.0;
    }

    /// Compute Gini coefficient based on TEND balance × price-weighted skill totals.
    ///
    /// Fix 3: Incorporates prices — agents producing scarce goods are "wealthier".
    /// Uses the sorted-list O(n log n) formula instead of the O(n^2) pairwise formula.
    /// Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    /// where x_i are sorted in ascending order and i is 1-indexed.
    pub fn compute_gini(&mut self, agents: &[CivAgent]) {
        let prices = self.prices;
        let mut totals: Vec<f64> = agents
            .iter()
            .filter(|a| a.is_alive())
            .map(|a| {
                let skills = a.skills.as_slice();
                let price_weighted: f64 =
                    skills.iter().enumerate().map(|(i, &s)| s * prices[i]).sum();
                // Blend TEND balance with price-weighted skills
                price_weighted + a.tend_balance.abs()
            })
            .collect();

        let n = totals.len();
        if n < 2 {
            self.gini_coefficient = 0.0;
            return;
        }

        totals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f64 = totals.iter().sum();
        if sum == 0.0 {
            self.gini_coefficient = 0.0;
            return;
        }

        let weighted_sum: f64 = totals
            .iter()
            .enumerate()
            .map(|(i, x)| (i + 1) as f64 * x)
            .sum();

        let nf = n as f64;
        self.gini_coefficient =
            ((2.0 * weighted_sum) / (nf * sum) - (nf + 1.0) / nf).clamp(0.0, 1.0);
    }

    /// Compute self-sufficiency from world resources.
    pub fn compute_self_sufficiency(&self, resources: &WorldResources) -> f64 {
        resources.self_sufficiency()
    }

    /// Invest a fraction of total production into infrastructure capital.
    ///
    /// `infrastructure_capital += total_production * fraction * 0.1`
    ///
    /// The 0.1 factor represents the conversion efficiency from production units
    /// to durable capital.
    pub fn invest(&mut self, fraction: f64) {
        let clamped = fraction.clamp(0.0, 0.5);
        self.investment_rate = clamped;
        self.infrastructure_capital += self.total_production * clamped * 0.1;
        // Cap at 100.0
        self.infrastructure_capital = self.infrastructure_capital.min(100.0);
    }

    /// GDP per capita (total production / population).
    pub fn gdp_per_capita(&self, population: usize) -> f64 {
        if population == 0 {
            return 0.0;
        }
        self.total_production / population as f64
    }

    /// Sector name for a given index.
    pub fn sector_name(index: usize) -> &'static str {
        if index < NUM_SECTORS {
            SKILL_SECTORS[index]
        } else {
            "unknown"
        }
    }

    /// Total number of assigned workers across all sectors.
    pub fn total_workers(&self) -> usize {
        self.sector_workers.iter().sum()
    }
}

impl Default for WorldEconomy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, ConsciousnessState, SkillVector};

    /// Reference tick used by all economy tests. Agents are born relative to this.
    const TEST_TICK: u32 = 1000;

    fn make_agent_with_skill(sector: usize, skill_level: f64, age_years: u32) -> CivAgent {
        let mut skills = SkillVector::new();
        skills.learn(sector, skill_level);
        let birth_tick = TEST_TICK - age_years * 12;
        CivAgent {
            id: 0,
            birth_tick,
            death_tick: None,
            sex: BiologicalSex::Male,
            world_id: 0,
            health: 1.0,
            skills,
            education_level: 0.5,
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
        }
    }

    #[test]
    fn test_production_scales_with_workers() {
        let mut econ = WorldEconomy::new();
        econ.sector_workers = [10, 0, 0, 0, 0, 0, 0, 0];
        econ.tick_production();
        let output_10 = econ.sector_output[0];

        econ.sector_workers = [20, 0, 0, 0, 0, 0, 0, 0];
        econ.tick_production();
        let output_20 = econ.sector_output[0];

        assert!(
            output_20 > output_10,
            "More workers should produce more: {output_20} vs {output_10}"
        );
    }

    #[test]
    fn test_tech_multiplier_increases_output() {
        let mut econ = WorldEconomy::new();
        econ.sector_workers = [10, 0, 0, 0, 0, 0, 0, 0];
        econ.tick_production();
        let base_output = econ.sector_output[0];

        econ.technology_multiplier[0] = 2.0;
        econ.tick_production();
        let boosted_output = econ.sector_output[0];

        assert!(
            (boosted_output - base_output * 2.0).abs() < 0.01,
            "2x tech should double output: {boosted_output} vs {base_output}"
        );
    }

    #[test]
    fn test_demurrage_reduces_supply() {
        let mut econ = WorldEconomy::new();
        let initial = econ.currency_supply;
        econ.tick_demurrage();
        assert!(
            econ.currency_supply < initial,
            "Demurrage should reduce supply: {} vs {initial}",
            econ.currency_supply
        );
        let expected = initial * (1.0 - 0.02 / 12.0);
        assert!(
            (econ.currency_supply - expected).abs() < 0.001,
            "Expected {expected}, got {}",
            econ.currency_supply
        );
    }

    #[test]
    fn test_gini_is_bounded() {
        let mut econ = WorldEconomy::new();

        // Identical agents -> Gini near 0
        let agents: Vec<CivAgent> = (0..20)
            .map(|i| {
                let mut a = make_agent_with_skill(0, 0.3, 30);
                a.id = i;
                a
            })
            .collect();
        econ.compute_gini(&agents);
        assert!(
            econ.gini_coefficient >= 0.0 && econ.gini_coefficient <= 1.0,
            "Gini should be in [0, 1], got {}",
            econ.gini_coefficient
        );
        assert!(
            econ.gini_coefficient < 0.1,
            "Identical agents should have low Gini, got {}",
            econ.gini_coefficient
        );

        // Very different agents -> higher Gini
        let mut diverse_agents = Vec::new();
        for i in 0..10u64 {
            let mut a = make_agent_with_skill(0, 0.9, 30);
            a.id = i;
            diverse_agents.push(a);
        }
        for i in 10..20u64 {
            let a = CivAgent {
                id: i,
                birth_tick: 0u32.wrapping_sub(30 * 12),
                death_tick: None,
                sex: BiologicalSex::Female,
                world_id: 0,
                health: 1.0,
                skills: SkillVector::new(),
                education_level: 0.0,
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
            diverse_agents.push(a);
        }
        econ.compute_gini(&diverse_agents);
        assert!(
            econ.gini_coefficient > 0.05,
            "Diverse agents should have higher Gini, got {}",
            econ.gini_coefficient
        );
    }

    #[test]
    fn test_self_sufficiency_calculation() {
        let econ = WorldEconomy::new();
        let resources = WorldResources::earth_default();
        let ss = econ.compute_self_sufficiency(&resources);
        assert!(ss > 0.9, "Earth should be near self-sufficient, got {ss}");

        let lunar = WorldResources::lunar_default();
        let ss_lunar = econ.compute_self_sufficiency(&lunar);
        assert!(ss_lunar < ss, "Lunar should be less self-sufficient");
    }

    #[test]
    fn test_investment_grows_infrastructure() {
        let mut econ = WorldEconomy::new();
        econ.sector_workers = [10; NUM_SECTORS];
        econ.tick_production();
        let infra_before = econ.infrastructure_capital;
        assert!(econ.total_production > 0.0);

        econ.invest(0.3);
        assert!(
            econ.infrastructure_capital > infra_before,
            "Investment should grow infrastructure: {} vs {infra_before}",
            econ.infrastructure_capital
        );
    }

    #[test]
    fn test_zero_workers_zero_output() {
        let mut econ = WorldEconomy::new();
        econ.sector_workers = [0; NUM_SECTORS];
        econ.tick_production();
        assert_eq!(econ.total_production, 0.0);
        for &o in &econ.sector_output {
            assert_eq!(o, 0.0);
        }
    }

    #[test]
    fn test_assign_workers_balances() {
        let mut econ = WorldEconomy::new();
        // 40 agents all skilled in engineering (sector 0)
        let agents: Vec<CivAgent> = (0..40)
            .map(|i| {
                let mut a = make_agent_with_skill(0, 0.9, 30);
                a.id = i as u64;
                a
            })
            .collect();

        econ.assign_workers(&agents, TEST_TICK);

        let total = econ.total_workers();
        assert_eq!(total, 40);
        let min_expected = ((40.0 * MIN_SECTOR_FRACTION).ceil() as usize).max(1);
        for (i, &count) in econ.sector_workers.iter().enumerate() {
            assert!(
                count >= min_expected,
                "Sector {} ({}) has {} workers, expected at least {min_expected}",
                i,
                WorldEconomy::sector_name(i),
                count
            );
        }
    }

    #[test]
    fn test_gdp_per_capita() {
        let mut econ = WorldEconomy::new();
        econ.total_production = 100.0;
        assert!((econ.gdp_per_capita(50) - 2.0).abs() < 0.001);
        assert_eq!(econ.gdp_per_capita(0), 0.0);
    }

    #[test]
    fn test_assign_workers_children_excluded() {
        let mut econ = WorldEconomy::new();
        let mut agents = Vec::new();
        // 10 children (age 5) should not be counted
        for i in 0..10u64 {
            let mut a = make_agent_with_skill(0, 0.5, 5);
            a.id = i;
            agents.push(a);
        }
        // 10 adults (age 30) should be counted
        for i in 10..20u64 {
            let mut a = make_agent_with_skill(1, 0.5, 30);
            a.id = i;
            agents.push(a);
        }
        econ.assign_workers(&agents, TEST_TICK);
        assert_eq!(econ.total_workers(), 10, "Only adults should be counted");
    }

    #[test]
    fn test_price_spikes_when_supply_drops() {
        // Fix 3: Price Formation
        let mut econ = WorldEconomy::new();
        // Normal production
        econ.sector_workers = [10; NUM_SECTORS];
        econ.tick_production();
        econ.tick_prices(80);
        let normal_price = econ.prices[0];

        // Drop sector 0 to zero output
        econ.sector_workers[0] = 0;
        econ.tick_production();
        econ.tick_prices(80);
        let spike_price = econ.prices[0];

        assert!(
            spike_price > normal_price,
            "Price should spike when supply drops: {spike_price} vs {normal_price}"
        );
    }

    #[test]
    fn test_production_uses_capital_exponent() {
        let mut econ = WorldEconomy::new();
        econ.sector_workers = [10, 0, 0, 0, 0, 0, 0, 0];

        // Low infrastructure
        econ.infrastructure_capital = 1.0;
        econ.tick_production();
        let low_cap = econ.sector_output[0];

        // High infrastructure
        econ.infrastructure_capital = 50.0;
        econ.tick_production();
        let high_cap = econ.sector_output[0];

        assert!(
            high_cap > low_cap,
            "Higher capital should produce more: {high_cap} vs {low_cap}"
        );
    }

    #[test]
    fn test_extended_production_with_scaling() {
        let mut econ = WorldEconomy::new();
        econ.sector_workers = [10, 10, 10, 10, 10, 10, 10, 10];

        // Baseline: no scaling
        econ.tick_production_extended(None, None);
        let baseline = econ.total_production;

        // With superlinear scaling (large population = more innovation per capita)
        let scaling_10k = ScalingFactors::compute(10000.0);
        econ.tick_production_extended(Some(&scaling_10k), None);
        let scaled = econ.total_production;

        assert!(
            scaled > baseline,
            "Superlinear scaling should boost production: {scaled} vs {baseline}"
        );
        // 10k pop at ref 1k => ratio 10, innovation mult = 10^(1/6) ≈ 1.468
        assert!(
            scaled > baseline * 1.3,
            "Should be at least 1.3x boost: {scaled} vs {} (baseline*1.3)",
            baseline * 1.3
        );
    }

    #[test]
    fn test_extended_production_with_energy() {
        let mut econ = WorldEconomy::new();
        econ.sector_workers = [10, 10, 10, 10, 10, 10, 10, 10];

        // Abundant energy
        econ.tick_production_extended(None, Some(200.0));
        let abundant = econ.total_production;

        // Scarce energy
        econ.tick_production_extended(None, Some(10.0));
        let scarce = econ.total_production;

        assert!(
            abundant > scarce,
            "More energy should produce more: {abundant} vs {scarce}"
        );
    }

    #[test]
    fn test_legacy_production_unchanged() {
        let mut econ1 = WorldEconomy::new();
        let mut econ2 = WorldEconomy::new();
        econ1.sector_workers = [15, 5, 5, 5, 5, 5, 5, 5];
        econ2.sector_workers = [15, 5, 5, 5, 5, 5, 5, 5];

        // Legacy path
        econ1.tick_production();
        // Extended path with None (should be identical)
        econ2.tick_production_extended(None, None);

        assert!(
            (econ1.total_production - econ2.total_production).abs() < 0.001,
            "Legacy and extended(None,None) should be identical: {} vs {}",
            econ1.total_production,
            econ2.total_production
        );
    }
}
