// SIMULATOR_ROADMAP A3 — attacker-dose sensitivity sweep.
//
// A2 found that Phase 2 machinery doesn't measurably outperform baseline
// at 3 attackers × 5 strategies = 15 adversaries. The null might be
// dose-specific: maybe the sim is under-attacked. This sweep ramps the
// dose and reports the CVS delta at each point.
//
// For each dose × phase2 ∈ {on, off}, runs 3 seeds × 50 years. Reports:
// - mean CVS per (dose, phase2) cell
// - delta (phase2=on − phase2=off) per dose
// - whether Phase 2 starts helping at some dose

use mycelix_multiworld_sim::config::{PolicyConfig, SimulationConfig};
use mycelix_multiworld_sim::red_team::AdversarialStrategy;
use mycelix_multiworld_sim::MultiWorldSimulator;

fn run(seed: u64, years: u32, phase2: bool, per_strategy: usize) -> (bool, f64) {
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = years * 12;
    config.seed = seed;
    config.policy = PolicyConfig::default();
    config.policy.phase2_enabled = phase2;

    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::TierBuyer, per_strategy);
    sim.inject_adversaries(AdversarialStrategy::DemurrageEvader, per_strategy);
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, per_strategy);
    sim.inject_adversaries(AdversarialStrategy::CrossClusterAmplifier, per_strategy);
    sim.inject_adversaries(AdversarialStrategy::GuildColluder, per_strategy);
    let report = sim.run();
    (report.survived, report.final_cvs)
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn main() {
    let seeds = [42u64, 137, 999];
    let years = 50u32;
    let doses = [3usize, 10, 20, 30];

    println!(
        "A3 dose-sensitivity — {} doses × {} seeds × {} years, Phase 2 on vs off",
        doses.len(),
        seeds.len(),
        years,
    );
    println!();
    println!(
        "{:>5} {:>8} {:>8} {:>9} {:>6} {:>6}",
        "dose", "cvs_on", "cvs_off", "delta", "surv_on", "surv_off",
    );
    println!("{}", "-".repeat(50));

    for &dose in &doses {
        let mut cvs_on = Vec::new();
        let mut cvs_off = Vec::new();
        let mut surv_on = 0usize;
        let mut surv_off = 0usize;
        for &seed in &seeds {
            let (so, co) = run(seed, years, true, dose);
            let (sf, cf) = run(seed, years, false, dose);
            cvs_on.push(co);
            cvs_off.push(cf);
            surv_on += so as usize;
            surv_off += sf as usize;
        }
        let m_on = mean(&cvs_on);
        let m_off = mean(&cvs_off);
        println!(
            "{:>5} {:>8.3} {:>8.3} {:>+9.3} {:>6} {:>6}",
            dose,
            m_on,
            m_off,
            m_on - m_off,
            surv_on,
            surv_off,
        );
    }
}
