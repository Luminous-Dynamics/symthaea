// Multi-seed A/B validation for the Mycelix red-team defenses.
//
// Runs N seeds × 50 years with 15 mixed attackers (3 of each of the 5
// Mycelix strategies). Prints per-seed metrics and aggregate stats so
// the single-seed claims from Phase 2c can be checked for robustness.

use mycelix_multiworld_sim::config::{PolicyConfig, SimulationConfig};
use mycelix_multiworld_sim::red_team::AdversarialStrategy;
use mycelix_multiworld_sim::MultiWorldSimulator;

#[derive(Debug, Clone, Copy)]
struct SeedResult {
    seed: u64,
    survived: bool,
    final_cvs: f64,
    final_pop: usize,
    farming_score: f64,
    farmer_rejected: u32,
    farmer_credited: u32,
    tier_buyer_delta: f64, // TierBuyer mean SAP − baseline mean SAP
    /// A1: mean of MycelixResilience over the 5 attack surfaces (NaN if
    /// the metric wasn't populated).
    resilience_mean: f64,
}

fn run_seed(seed: u64, years: u32) -> SeedResult {
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = years * 12;
    config.seed = seed;
    config.policy = PolicyConfig::default();

    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::TierBuyer, 3);
    sim.inject_adversaries(AdversarialStrategy::DemurrageEvader, 3);
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, 3);
    sim.inject_adversaries(AdversarialStrategy::CrossClusterAmplifier, 3);
    sim.inject_adversaries(AdversarialStrategy::GuildColluder, 3);
    let report = sim.run();

    let mut farmer_credited = 0u32;
    let mut farmer_rejected = 0u32;
    let mut buyer_sap = 0.0;
    let mut buyer_count = 0usize;
    let mut other_sap = 0.0;
    let mut other_count = 0usize;

    for world in &sim.worlds {
        for a in world.agents.iter().filter(|a| a.is_alive()) {
            match a.adversarial {
                Some(AdversarialStrategy::CorrectionFarmer) => {
                    farmer_credited += a.justice.corrections;
                    farmer_rejected += a.justice.rejected_corrections;
                }
                Some(AdversarialStrategy::TierBuyer) => {
                    buyer_sap += a.sap_balance;
                    buyer_count += 1;
                }
                _ => {
                    other_sap += a.sap_balance;
                    other_count += 1;
                }
            }
        }
    }

    let farming_score = if farmer_credited + farmer_rejected == 0 {
        0.0
    } else {
        farmer_rejected as f64 / (farmer_credited + farmer_rejected) as f64
    };
    let tier_buyer_delta = if buyer_count > 0 && other_count > 0 {
        buyer_sap / buyer_count as f64 - other_sap / other_count as f64
    } else {
        0.0
    };

    let resilience_mean = report
        .mycelix_resilience
        .as_ref()
        .map(|r| r.mean())
        .unwrap_or(f64::NAN);

    SeedResult {
        seed,
        survived: report.survived,
        final_cvs: report.final_cvs,
        final_pop: report.final_population,
        farming_score,
        farmer_rejected,
        farmer_credited,
        tier_buyer_delta,
        resilience_mean,
    }
}

fn main() {
    let seeds = [7u64, 13, 42, 101, 137, 271, 314, 577, 999, 2024];
    let years = 50u32;

    println!(
        "Multi-seed A/B sweep — {} seeds × {} years with mixed Mycelix attack",
        seeds.len(),
        years,
    );
    println!();
    println!(
        "{:>6} {:>8} {:>8} {:>7} {:>9} {:>10} {:>10} {:>10} {:>10}",
        "seed",
        "survive",
        "cvs",
        "pop",
        "farm_scr",
        "f_rejected",
        "f_credited",
        "tb_delta",
        "resilnc",
    );
    println!("{}", "-".repeat(93));

    let mut results = Vec::new();
    for &seed in &seeds {
        let r = run_seed(seed, years);
        println!(
            "{:>6} {:>8} {:>8.3} {:>7} {:>9.3} {:>10} {:>10} {:>+10.2} {:>10.3}",
            r.seed,
            r.survived,
            r.final_cvs,
            r.final_pop,
            r.farming_score,
            r.farmer_rejected,
            r.farmer_credited,
            r.tier_buyer_delta,
            r.resilience_mean,
        );
        results.push(r);
    }

    // Aggregates.
    let n = results.len() as f64;
    let survival_rate = results.iter().filter(|r| r.survived).count() as f64 / n;
    let mean_cvs = results.iter().map(|r| r.final_cvs).sum::<f64>() / n;
    let var_cvs = results
        .iter()
        .map(|r| (r.final_cvs - mean_cvs).powi(2))
        .sum::<f64>()
        / n;
    let std_cvs = var_cvs.sqrt();
    let min_cvs = results
        .iter()
        .map(|r| r.final_cvs)
        .fold(f64::INFINITY, f64::min);
    let max_cvs = results
        .iter()
        .map(|r| r.final_cvs)
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_farming = results.iter().map(|r| r.farming_score).sum::<f64>() / n;
    let min_farming = results
        .iter()
        .map(|r| r.farming_score)
        .fold(f64::INFINITY, f64::min);

    println!();
    println!("Aggregate:");
    println!("  survival_rate = {:.0}%", survival_rate * 100.0);
    println!(
        "  CVS           mean {:.3} ± {:.3}, min {:.3}, max {:.3}",
        mean_cvs, std_cvs, min_cvs, max_cvs,
    );
    println!(
        "  farming_score mean {:.3}, min {:.3}",
        mean_farming, min_farming,
    );
    let mean_resilience = results
        .iter()
        .map(|r| r.resilience_mean)
        .filter(|x| x.is_finite())
        .sum::<f64>()
        / n;
    println!("  resilience    mean {:.3}", mean_resilience);
}
