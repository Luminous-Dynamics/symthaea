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
use crate::world::WorldResources;

/// Number of economic sectors (matches [`SKILL_SECTORS`]).
pub const NUM_SECTORS: usize = 8;

/// Labor share exponent in Cobb-Douglas production (alpha).
const LABOR_EXPONENT: f64 = 0.7;

/// Capital share exponent in Cobb-Douglas production (1 - alpha).
const CAPITAL_EXPONENT: f64 = 0.3;

/// Minimum workforce fraction per sector when population > 20.
const MIN_SECTOR_FRACTION: f64 = 0.05;

/// Default annual demurrage rate (2%).
const DEFAULT_DEMURRAGE_RATE: f64 = 0.02;

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
        }
    }

    /// Assign workers to sectors based on their strongest skill.
    ///
    /// Each working-age agent is placed in the sector matching their highest skill.
    /// If population > 20, a minimum of 5% of workers is guaranteed per sector
    /// by reassigning excess workers from over-represented sectors.
    pub fn assign_workers(&mut self, agents: &[CivAgent], current_tick: u32) {
        self.sector_workers = [0; NUM_SECTORS];

        // Count workers by strongest skill
        let mut worker_count = 0usize;
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
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.sector_workers[best_sector] += 1;
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
    /// `output_i = tech_mult_i * (effective_labor_i)^0.7 * infrastructure^0.3`
    ///
    /// Labor exponent is 0.7 (labor-intensive early colony). Capital exponent is 0.3.
    pub fn tick_production(&mut self) {
        self.total_production = 0.0;

        for i in 0..NUM_SECTORS {
            let workers = self.sector_workers[i] as f64;
            if workers == 0.0 {
                self.sector_output[i] = 0.0;
                continue;
            }

            let effective_labor = workers * self.labor_productivity[i];
            let output = self.technology_multiplier[i]
                * effective_labor.powf(LABOR_EXPONENT)
                * self.infrastructure_capital.max(0.01).powf(CAPITAL_EXPONENT);

            self.sector_output[i] = output;
            self.total_production += output;
        }
    }

    /// Apply monthly demurrage to the currency supply.
    ///
    /// `currency_supply *= (1.0 - demurrage_rate / 12.0)`
    pub fn tick_demurrage(&mut self) {
        self.currency_supply *= 1.0 - self.demurrage_rate / 12.0;
    }

    /// Compute Gini coefficient based on skill totals as a proxy for wealth.
    ///
    /// Uses the sorted-list O(n log n) formula instead of the O(n^2) pairwise formula.
    /// Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    /// where x_i are sorted in ascending order and i is 1-indexed.
    pub fn compute_gini(&mut self, agents: &[CivAgent]) {
        let mut totals: Vec<f64> = agents
            .iter()
            .filter(|a| a.is_alive())
            .map(|a| a.skills.total())
            .collect();

        let n = totals.len();
        if n < 2 {
            self.gini_coefficient = 0.0;
            return;
        }

        totals.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
}
