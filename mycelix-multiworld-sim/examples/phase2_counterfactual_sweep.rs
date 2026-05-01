// SIMULATOR_ROADMAP A2 — counterfactual sweep.
//
// For each seed, run the same mixed Mycelix attack twice: once with
// Phase 2 defenses enabled (default), once with them disabled (baseline
// scalar Phi + MYCEL governance that predated Phase 2a). Print the
// per-seed CVS delta and an aggregate.
//
// This answers the scientific question: does the Phase 2 machinery
// actually improve outcomes vs. the older baseline under the same
// adversarial pressure?

use mycelix_multiworld_sim::config::{PolicyConfig, SimulationConfig};
use mycelix_multiworld_sim::red_team::AdversarialStrategy;
use mycelix_multiworld_sim::MultiWorldSimulator;

#[derive(Debug, Clone, Copy)]
struct PairResult {
    seed: u64,
    cvs_on: f64,
    cvs_off: f64,
    cvs_geo_on: f64,
    cvs_geo_off: f64,
    pop_on: usize,
    pop_off: usize,
    res_on: f64,
    res_off: f64,
}

fn run_condition(seed: u64, years: u32, phase2: bool) -> (f64, f64, usize, f64) {
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = years * 12;
    config.seed = seed;
    config.policy = PolicyConfig::default();
    config.policy.phase2_enabled = phase2;

    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::TierBuyer, 3);
    sim.inject_adversaries(AdversarialStrategy::DemurrageEvader, 3);
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, 3);
    sim.inject_adversaries(AdversarialStrategy::CrossClusterAmplifier, 3);
    sim.inject_adversaries(AdversarialStrategy::GuildColluder, 3);
    let report = sim.run();
    let res = report
        .mycelix_resilience
        .as_ref()
        .map(|r| r.mean())
        .unwrap_or(f64::NAN);
    (
        report.final_cvs,
        report.final_cvs_geometric,
        report.final_population,
        res,
    )
}

fn main() {
    let seeds = [7u64, 13, 42, 101, 137, 271, 314, 577, 999, 2024];
    let years = 50u32;

    println!(
        "A2 counterfactual — {} seeds × {} years, Phase 2 enabled vs disabled",
        seeds.len(),
        years,
    );
    println!();
    println!(
        "{:>6} {:>7} {:>7} {:>+8} {:>7} {:>7} {:>+8} {:>7} {:>7}",
        "seed", "cvs_on", "cvs_off", "d_arith", "geo_on", "geo_off", "d_geo", "res_on", "res_off",
    );
    println!("{}", "-".repeat(79));

    let mut results = Vec::new();
    for &seed in &seeds {
        let (cvs_on, geo_on, pop_on, res_on) = run_condition(seed, years, true);
        let (cvs_off, geo_off, pop_off, res_off) = run_condition(seed, years, false);
        println!(
            "{:>6} {:>7.3} {:>7.3} {:>+8.3} {:>7.3} {:>7.3} {:>+8.3} {:>7.3} {:>7.3}",
            seed,
            cvs_on,
            cvs_off,
            cvs_on - cvs_off,
            geo_on,
            geo_off,
            geo_on - geo_off,
            res_on,
            res_off,
        );
        results.push(PairResult {
            seed,
            cvs_on,
            cvs_off,
            cvs_geo_on: geo_on,
            cvs_geo_off: geo_off,
            pop_on,
            pop_off,
            res_on,
            res_off,
        });
    }

    let n = results.len() as f64;
    let mean_d_arith = results.iter().map(|r| r.cvs_on - r.cvs_off).sum::<f64>() / n;
    let std_d_arith = (results
        .iter()
        .map(|r| (r.cvs_on - r.cvs_off - mean_d_arith).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    let mean_d_geo = results
        .iter()
        .map(|r| r.cvs_geo_on - r.cvs_geo_off)
        .sum::<f64>()
        / n;
    let std_d_geo = (results
        .iter()
        .map(|r| (r.cvs_geo_on - r.cvs_geo_off - mean_d_geo).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    let wins_arith = results.iter().filter(|r| r.cvs_on > r.cvs_off).count();
    let wins_geo = results
        .iter()
        .filter(|r| r.cvs_geo_on > r.cvs_geo_off)
        .count();
    let mean_res_on = results
        .iter()
        .map(|r| r.res_on)
        .filter(|x| x.is_finite())
        .sum::<f64>()
        / n;
    let mean_res_off = results
        .iter()
        .map(|r| r.res_off)
        .filter(|x| x.is_finite())
        .sum::<f64>()
        / n;

    println!();
    println!("Aggregate:");
    println!(
        "  Arithmetic CVS delta  mean {:+.3} ± {:.3}, Phase 2 wins {}/{}",
        mean_d_arith,
        std_d_arith,
        wins_arith,
        seeds.len(),
    );
    println!(
        "  Geometric  CVS delta  mean {:+.3} ± {:.3}, Phase 2 wins {}/{}",
        mean_d_geo,
        std_d_geo,
        wins_geo,
        seeds.len(),
    );
    println!(
        "  Resilience            phase2=on {:.3} vs phase2=off {:.3}",
        mean_res_on, mean_res_off,
    );
}
