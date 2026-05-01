// Quick metrics dump for the Phase 2c attack wiring.
// Runs a 50-year mixed-attack sim and prints defense telemetry.
//
// Reference numbers (seed 42):
//   5 year:  survived, CVS 0.670, farming 0.612, TierBuyer +0.35 SAP vs pop
//   50 year: survived, CVS 0.726, farming 0.612, TierBuyer −2.78 SAP vs pop
// The negative delta at 50 years is the dilution signature — injected
// adversaries don't procreate attackers, so general population outgrows the
// one-shot attack over generations.

use mycelix_multiworld_sim::config::{PolicyConfig, SimulationConfig};
use mycelix_multiworld_sim::red_team::AdversarialStrategy;
use mycelix_multiworld_sim::MultiWorldSimulator;

fn main() {
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = 50 * 12;
    config.seed = 42;
    config.policy = PolicyConfig::default();

    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::TierBuyer, 3);
    sim.inject_adversaries(AdversarialStrategy::DemurrageEvader, 3);
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, 3);
    sim.inject_adversaries(AdversarialStrategy::CrossClusterAmplifier, 3);
    sim.inject_adversaries(AdversarialStrategy::GuildColluder, 3);
    let report = sim.run();

    println!("survived:       {}", report.survived);
    println!("final_cvs:      {:.3}", report.final_cvs);
    println!("final_phi:      {:.3}", report.final_collective_phi);
    println!("final_pop:      {}", report.final_population);

    let mut farmers_credited = 0u32;
    let mut farmers_rejected = 0u32;
    let mut buyer_sap_sum = 0.0;
    let mut buyer_count = 0usize;
    let mut other_sap_sum = 0.0;
    let mut other_count = 0usize;

    for world in &sim.worlds {
        for a in world.agents.iter().filter(|a| a.is_alive()) {
            match a.adversarial {
                Some(AdversarialStrategy::CorrectionFarmer) => {
                    farmers_credited += a.justice.corrections;
                    farmers_rejected += a.justice.rejected_corrections;
                }
                Some(AdversarialStrategy::TierBuyer) => {
                    buyer_sap_sum += a.sap_balance;
                    buyer_count += 1;
                }
                _ => {
                    other_sap_sum += a.sap_balance;
                    other_count += 1;
                }
            }
        }
    }

    let farm_score = if farmers_credited + farmers_rejected == 0 {
        0.0
    } else {
        farmers_rejected as f64 / (farmers_credited + farmers_rejected) as f64
    };
    println!(
        "farmer:         credited={} rejected={} farming_score={:.3}",
        farmers_credited, farmers_rejected, farm_score,
    );
    if buyer_count > 0 && other_count > 0 {
        let buyer_mean = buyer_sap_sum / buyer_count as f64;
        let other_mean = other_sap_sum / other_count as f64;
        println!(
            "tier_buyer:     mean_sap={:.2} vs baseline_mean={:.2} delta={:+.2}",
            buyer_mean,
            other_mean,
            buyer_mean - other_mean,
        );
    }
}
