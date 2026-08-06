// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Genesis v0 exploratory run, per `ALIFE_MULTIAGENT_GENESIS_PLAN_2026-07-25.md`'s "Workflow
//! discipline": **exploratory, not confirmatory**. Many seeds, watched, no pre-registered
//! hypothesis, no statistical claim. Prints descriptive summaries only. Any pattern worth
//! pursuing should be written up as "Candidate phenomenon: ..." (in a report, not asserted here
//! as a test) and become its own preregistered MA-00N experiment before any claim about
//! cooperation/reciprocity/trust is made.
//!
//! Revised for the "Genesis v0.1 -- Causal Plumbing Audit" (Gate 1): the original run's "0A"
//! only disabled mutation while leaving reproduction fully enabled, conflating "no mutation"
//! with "no reproduction." Three explicit reproduction arms now exist, and paired/unpaired tick
//! fractions are reported per condition (Gate 2's measurement requirement) so any future
//! comparison can see, not silently absorb, differences in social exposure between pairing modes.
//!
//! Run: `cargo run -p symthaea-alife --example genesis_explore`

use symthaea_alife::{
    EncounterScheduler, OrganismConfig, PairingMode, Population, PopulationConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReproductionArm {
    /// Reproduction itself disabled (an unreachable threshold, matching the test suite's
    /// `no_churn_population_cfg` pattern) -- the *actual* lifetime-only condition G0f specifies.
    /// Population and identities are fixed for the whole run.
    LifetimeOnly,
    /// Reproduction enabled, mutation disabled -- offspring are exact genome copies. Isolates
    /// "does turnover/selection alone matter" from "does heritable variation matter."
    Clonal,
    /// Reproduction and mutation both enabled -- the original "0B."
    Evolution,
}

impl ReproductionArm {
    fn label(self) -> &'static str {
        match self {
            ReproductionArm::LifetimeOnly => "Lifetime",
            ReproductionArm::Clonal => "Clonal",
            ReproductionArm::Evolution => "Evolution",
        }
    }

    fn reproduction_energy_threshold(self) -> f64 {
        match self {
            // Energy is clamped to [0, 1], so 10.0 can never be reached -- reproduction is
            // genuinely, structurally disabled, not just made rare.
            ReproductionArm::LifetimeOnly => 10.0,
            ReproductionArm::Clonal | ReproductionArm::Evolution => 0.85,
        }
    }

    fn mutation_rate(self) -> f64 {
        match self {
            ReproductionArm::LifetimeOnly | ReproductionArm::Clonal => 0.0,
            ReproductionArm::Evolution => 0.1,
        }
    }
}

fn social_cfg(transfer_quantum: f64) -> OrganismConfig {
    OrganismConfig {
        social_enabled: true,
        transfer_quantum,
        forage_efficiency: 0.6, // Phase 1's "sustainable" value
        ..OrganismConfig::default()
    }
}

struct RunSummary {
    arm: &'static str,
    pairing: PairingMode,
    seed: u64,
    initial_population: usize,
    final_population: usize,
    total_births: u64,
    total_deaths: u64,
    paired_frac: f64,
    forage_frac: f64,
    rest_frac: f64,
    transfer_frac: f64,
    total_transferred: f64,
    avg_distinct_partners: f64,
    max_encounter_count: u32,
    max_single_partner_given: f64,
}

fn run_condition(arm: ReproductionArm, pairing: PairingMode, seed: u64, ticks: u64) -> RunSummary {
    let initial_population = 16;
    let population_cfg = PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: arm.reproduction_energy_threshold(),
        reproduction_energy_cost: 0.4,
        organism_cfg: social_cfg(0.05),
        mutation_rate: arm.mutation_rate(),
        mutation_std: 0.05,
        ..Default::default()
    };
    let mut pop = Population::new(population_cfg, initial_population, seed);
    let mut sched = EncounterScheduler::new(pairing, seed.wrapping_add(9_999));

    let mut forage = 0u64;
    let mut rest = 0u64;
    let mut transfer = 0u64;
    let mut paired = 0u64;
    let mut total_transferred = 0.0;
    let mut total_organism_ticks = 0u64;

    for _ in 0..ticks {
        if pop.is_empty() {
            break;
        }
        // Density-divided resource share -- NOT a flat constant. A flat share caused a real
        // unbounded-population-growth incident during this plan's Stage 0 test development.
        pop.step_social(|n| 4.0 / (n.max(1) as f64), &mut sched);
        for e in pop.drain_event_log() {
            total_organism_ticks += 1;
            if e.partner_id.is_some() {
                paired += 1;
            }
            match e.action {
                0 => forage += 1,
                1 => rest += 1,
                _ => transfer += 1,
            }
            total_transferred += e.transfer_amount;
        }
    }

    let avg_distinct_partners = if pop.is_empty() {
        0.0
    } else {
        pop.organisms
            .iter()
            .map(|o| o.ledger.len() as f64)
            .sum::<f64>()
            / pop.len() as f64
    };
    let max_encounter_count = pop
        .organisms
        .iter()
        .flat_map(|o| o.ledger.values())
        .map(|r| r.encounter_count)
        .max()
        .unwrap_or(0);
    let max_single_partner_given = pop
        .organisms
        .iter()
        .flat_map(|o| o.ledger.values())
        .map(|r| r.given_to_partner)
        .fold(0.0f64, f64::max);

    RunSummary {
        arm: arm.label(),
        pairing,
        seed,
        initial_population,
        final_population: pop.len(),
        total_births: pop.total_births,
        total_deaths: pop.total_deaths,
        paired_frac: paired as f64 / total_organism_ticks.max(1) as f64,
        forage_frac: forage as f64 / total_organism_ticks.max(1) as f64,
        rest_frac: rest as f64 / total_organism_ticks.max(1) as f64,
        transfer_frac: transfer as f64 / total_organism_ticks.max(1) as f64,
        total_transferred,
        avg_distinct_partners,
        max_encounter_count,
        max_single_partner_given,
    }
}

fn print_summary(s: &RunSummary) {
    println!(
        "{:>9} {:14?} seed={:2} | pop {:3}->{:3} births={:4} deaths={:4} | paired={:.3} forage={:.3} rest={:.3} transfer={:.3} | given_total={:7.2} avg_partners={:4.1} max_enc={:4} max_single_given={:.2}",
        s.arm,
        s.pairing,
        s.seed,
        s.initial_population,
        s.final_population,
        s.total_births,
        s.total_deaths,
        s.paired_frac,
        s.forage_frac,
        s.rest_frac,
        s.transfer_frac,
        s.total_transferred,
        s.avg_distinct_partners,
        s.max_encounter_count,
        s.max_single_partner_given,
    );
}

fn main() {
    println!(
        "Genesis v0.1 exploratory run -- descriptive only, no hypothesis, no statistical claim.\n"
    );

    let seeds: [u64; 5] = [1, 2, 3, 4, 5];
    let ticks: u64 = 2000;

    for arm in [
        ReproductionArm::LifetimeOnly,
        ReproductionArm::Clonal,
        ReproductionArm::Evolution,
    ] {
        println!("=== {} ===", arm.label());
        for pairing in [PairingMode::Random, PairingMode::FixedPartners] {
            for &seed in &seeds {
                print_summary(&run_condition(arm, pairing, seed, ticks));
            }
        }
        println!();
    }
}
