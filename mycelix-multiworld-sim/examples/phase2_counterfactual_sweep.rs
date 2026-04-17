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

use mycelix_multiworld_sim::MultiWorldSimulator;
use mycelix_multiworld_sim::config::{PolicyConfig, SimulationConfig};
use mycelix_multiworld_sim::red_team::AdversarialStrategy;

#[derive(Debug, Clone, Copy)]
struct PairResult {
    seed: u64,
    cvs_on: f64,
    cvs_off: f64,
    pop_on: usize,
    pop_off: usize,
    res_on: f64,
    res_off: f64,
}

fn run_condition(seed: u64, years: u32, phase2: bool) -> (f64, usize, f64) {
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
    (report.final_cvs, report.final_population, res)
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
        "{:>6} {:>8} {:>8} {:>9} {:>8} {:>8} {:>8} {:>8}",
        "seed", "cvs_on", "cvs_off", "d_cvs", "pop_on", "pop_off", "res_on", "res_off",
    );
    println!("{}", "-".repeat(76));

    let mut results = Vec::new();
    for &seed in &seeds {
        let (cvs_on, pop_on, res_on) = run_condition(seed, years, true);
        let (cvs_off, pop_off, res_off) = run_condition(seed, years, false);
        let delta = cvs_on - cvs_off;
        println!(
            "{:>6} {:>8.3} {:>8.3} {:>+9.3} {:>8} {:>8} {:>8.3} {:>8.3}",
            seed, cvs_on, cvs_off, delta, pop_on, pop_off, res_on, res_off,
        );
        results.push(PairResult {
            seed,
            cvs_on,
            cvs_off,
            pop_on,
            pop_off,
            res_on,
            res_off,
        });
    }

    let n = results.len() as f64;
    let mean_delta = results.iter().map(|r| r.cvs_on - r.cvs_off).sum::<f64>() / n;
    let var_delta = results
        .iter()
        .map(|r| {
            let d = r.cvs_on - r.cvs_off;
            (d - mean_delta).powi(2)
        })
        .sum::<f64>()
        / n;
    let std_delta = var_delta.sqrt();
    let wins = results.iter().filter(|r| r.cvs_on > r.cvs_off).count();
    let losses = results.iter().filter(|r| r.cvs_on < r.cvs_off).count();
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
        "  CVS delta     mean {:+.3} ± {:.3} (phase2=on − phase2=off)",
        mean_delta, std_delta,
    );
    println!(
        "  Phase 2 wins  {}/{} seeds (on > off), losses {}, ties {}",
        wins,
        seeds.len(),
        losses,
        seeds.len() - wins - losses,
    );
    println!(
        "  Resilience    phase2=on {:.3} vs phase2=off {:.3}",
        mean_res_on, mean_res_off,
    );
}
