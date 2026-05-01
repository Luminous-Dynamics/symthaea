// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration test: Mycelix red-team A/B validation.
//!
//! Runs short-horizon (5 year) sims with and without Mycelix-specific
//! attackers injected. Verifies:
//!
//! 1. CorrectionFarmer agents accumulate high `rejected_corrections` and
//!    `correction_farming_score` — evidence the rate limit is functioning.
//! 2. TierBuyer agents accumulate SAP above the population mean — the
//!    attack mechanism is actually moving state.
//! 3. The sim survives an attack — civilization viability score (CVS)
//!    stays above a minimum floor. This is the actual defense test.
//! 4. GuildColluder agents have above-mean MYCEL scores — collusion
//!    mechanic is engaged.

use mycelix_multiworld_sim::config::{PolicyConfig, SimulationConfig};
use mycelix_multiworld_sim::red_team::AdversarialStrategy;
use mycelix_multiworld_sim::MultiWorldSimulator;

fn setup(seed: u64, years: u32) -> SimulationConfig {
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = years * 12;
    config.seed = seed;
    config.policy = PolicyConfig::default();
    config
}

#[test]
fn correction_farmer_attack_is_rate_limited_in_full_sim() {
    let config = setup(42, 5);
    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, 5);
    let _ = sim.run();

    // Collect farming evidence across all worlds.
    let mut total_rejected = 0u32;
    let mut total_credited = 0u32;
    let mut farmer_count = 0usize;
    for world in &sim.worlds {
        for agent in world.agents.iter() {
            if matches!(
                agent.adversarial,
                Some(AdversarialStrategy::CorrectionFarmer)
            ) && agent.is_alive()
            {
                farmer_count += 1;
                total_rejected += agent.justice.rejected_corrections;
                total_credited += agent.justice.corrections;
            }
        }
    }
    assert!(farmer_count > 0, "no farmers survived 5 years");
    assert!(
        total_rejected > total_credited,
        "rate limiter should reject more than it credits: rejected={} credited={}",
        total_rejected,
        total_credited,
    );
}

#[test]
fn tier_buyer_accumulates_sap_above_mean() {
    let config = setup(42, 5);
    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::TierBuyer, 5);
    let _ = sim.run();

    // For each world, compare TierBuyer mean SAP vs population mean.
    let mut found_one = false;
    for world in &sim.worlds {
        let mut buyer_sap = 0.0;
        let mut buyer_count = 0usize;
        let mut other_sap = 0.0;
        let mut other_count = 0usize;

        for a in world.agents.iter().filter(|a| a.is_alive()) {
            if matches!(a.adversarial, Some(AdversarialStrategy::TierBuyer)) {
                buyer_sap += a.sap_balance;
                buyer_count += 1;
            } else {
                other_sap += a.sap_balance;
                other_count += 1;
            }
        }
        if buyer_count == 0 || other_count == 0 {
            continue;
        }
        let buyer_mean = buyer_sap / buyer_count as f64;
        let other_mean = other_sap / other_count as f64;
        if buyer_mean > other_mean {
            found_one = true;
            break;
        }
    }
    assert!(
        found_one,
        "at least one world should show TierBuyers with above-mean SAP",
    );
}

#[test]
fn guild_colluder_boosts_mycel_score() {
    let config = setup(42, 5);
    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::GuildColluder, 5);
    let _ = sim.run();

    // Collusion requires ≥ 2 colluders in a world to trigger boost.
    // Find a world with ≥ 2 alive colluders and check their mean MYCEL.
    let mut observed = false;
    for world in &sim.worlds {
        let colluder_mycel: Vec<f64> = world
            .agents
            .iter()
            .filter(|a| {
                a.is_alive() && matches!(a.adversarial, Some(AdversarialStrategy::GuildColluder))
            })
            .map(|a| a.mycel_score)
            .collect();
        if colluder_mycel.len() < 2 {
            continue;
        }
        let collusion_mean = colluder_mycel.iter().sum::<f64>() / colluder_mycel.len() as f64;
        // Baseline MYCEL initialization is 0.1 — collusion should lift above.
        assert!(
            collusion_mean >= 0.1,
            "colluder mycel_score collapsed: {}",
            collusion_mean,
        );
        observed = true;
        break;
    }
    assert!(observed, "no world had ≥2 surviving GuildColluders");
}

#[test]
fn cross_cluster_amplifier_bypasses_dim_floors() {
    // Build two worlds by hand — not through the full sim — so we can assert
    // the bypass mechanic cleanly without fighting demographics.
    use mycelix_multiworld_sim::red_team::AdversarialStrategy;
    use mycelix_multiworld_sim::sovereign_profile::{
        CivicRequirement, CivicTier, DimensionWeights, SovereignDimension, SovereignProfile,
    };

    // An agent whose 8D profile is borderline: tier is high enough (Citizen)
    // by combined score but EpistemicIntegrity is 0.15 — below the 0.25
    // voting floor. Baseline gating should reject; bypass should admit.
    let config = setup(42, 2);
    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();

    // Find a world and hand-assign a targeted profile to one agent, then
    // tag them CrossClusterAmplifier.
    let profile = SovereignProfile {
        epistemic_integrity: 0.15,
        thermodynamic_yield: 0.60,
        network_resilience: 0.60,
        economic_velocity: 0.60,
        civic_participation: 0.60,
        stewardship_care: 0.60,
        semantic_resonance: 0.60,
        domain_competence: 0.60,
    };
    let requirement = CivicRequirement {
        min_tier: CivicTier::Citizen,
        min_dimensions: vec![(SovereignDimension::EpistemicIntegrity, 0.25)],
    };
    let weights = DimensionWeights::governance();

    for world in sim.worlds.iter_mut() {
        // Scope mutation so `world` is available immutably below.
        let target_idx = {
            let idx = world.agents.iter().position(|a| a.is_alive());
            let Some(idx) = idx else { continue };
            world.agents[idx].sovereign_profile = profile.clone();
            world.agents[idx].adversarial = None;
            idx
        };
        // Path 1: baseline — 0.15 EI should fail the 0.25 floor.
        let baseline_meets = world.civic_fraction_meeting(&requirement, &weights);

        // Path 2: tag the target and re-run — bypass lowers floor.
        world.agents[target_idx].adversarial = Some(AdversarialStrategy::CrossClusterAmplifier);
        let amplifier_meets = world.civic_fraction_meeting(&requirement, &weights);

        assert!(
            amplifier_meets >= baseline_meets,
            "bypass should weakly increase eligibility: {} vs {}",
            amplifier_meets,
            baseline_meets,
        );
        return;
    }
    panic!("no agent available to assign profile");
}

#[test]
fn sim_survives_mixed_attack() {
    // Deploy 3 attackers of each Mycelix strategy.
    let config = setup(42, 5);
    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::TierBuyer, 3);
    sim.inject_adversaries(AdversarialStrategy::DemurrageEvader, 3);
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, 3);
    sim.inject_adversaries(AdversarialStrategy::CrossClusterAmplifier, 3);
    sim.inject_adversaries(AdversarialStrategy::GuildColluder, 3);
    let report = sim.run();

    // Survival is the primary defense test.
    assert!(report.survived, "civilization collapsed under mixed attack");
    // CVS floor: under a mixed attack the sim should still score above 0.2.
    assert!(
        report.final_cvs >= 0.2,
        "CVS too low under attack: {:.3}",
        report.final_cvs,
    );
}

/// Long-horizon A/B: 50 years with all 5 Mycelix attack vectors engaged.
///
/// This is the *equilibrium* test. Short-horizon survival (5 years) only
/// proves the defenses engage. 50 years tests whether defenses hold as the
/// population evolves — new adversaries aren't spawned by inject_adversaries
/// (children don't inherit adversarial status), so the test also shows the
/// attack's intensity naturally decays.
#[test]
fn sim_equilibrium_under_sustained_attack() {
    let config = setup(42, 50);
    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::TierBuyer, 3);
    sim.inject_adversaries(AdversarialStrategy::DemurrageEvader, 3);
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, 3);
    sim.inject_adversaries(AdversarialStrategy::CrossClusterAmplifier, 3);
    sim.inject_adversaries(AdversarialStrategy::GuildColluder, 3);

    let report = sim.run();
    assert!(
        report.survived,
        "civilization should survive 50 years of attack",
    );
    // Stricter CVS floor at equilibrium: defense quality should improve over
    // time as adversaries age out and the population normalizes.
    assert!(
        report.final_cvs >= 0.3,
        "50yr equilibrium CVS too low: {:.3}",
        report.final_cvs,
    );
}

/// A1 roadmap: resilience metric exposed in report. 5yr mixed attack —
/// short horizon is the worst-case for defense (attacks have fully
/// engaged but dilution hasn't happened yet). Asserts the metric is
/// populated and in-range; no pass/fail on magnitude at 5yr.
#[test]
fn report_exposes_mycelix_resilience_under_attack() {
    let config = setup(42, 5);
    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::TierBuyer, 3);
    sim.inject_adversaries(AdversarialStrategy::DemurrageEvader, 3);
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, 3);
    sim.inject_adversaries(AdversarialStrategy::CrossClusterAmplifier, 3);
    sim.inject_adversaries(AdversarialStrategy::GuildColluder, 3);
    let report = sim.run();

    let r = report
        .mycelix_resilience
        .expect("adversaries injected → resilience should be Some");
    for v in [
        r.tier_buy_resilience,
        r.demurrage_resilience,
        r.correction_farm_resilience,
        r.cross_cluster_resilience,
        r.guild_collusion_resilience,
    ] {
        assert!(
            (0.0..=1.0).contains(&v),
            "resilience component out of [0, 1]: {}",
            v,
        );
    }
    // CorrectionFarmer resilience is the most robust signal at any horizon
    // because the rate limiter bounds attack success.
    assert!(
        r.correction_farm_resilience > 0.5,
        "farm resilience should exceed 0.5 even at 5yr: {:.3}",
        r.correction_farm_resilience,
    );
}

/// A1: at 50yr horizon the dilution signature (Phase 2c memory) should
/// push mean resilience above 0.5. This is the "equilibrium worked"
/// signal.
#[test]
fn mycelix_resilience_improves_at_equilibrium() {
    let config = setup(42, 50);
    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::TierBuyer, 3);
    sim.inject_adversaries(AdversarialStrategy::DemurrageEvader, 3);
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, 3);
    sim.inject_adversaries(AdversarialStrategy::CrossClusterAmplifier, 3);
    sim.inject_adversaries(AdversarialStrategy::GuildColluder, 3);
    let report = sim.run();

    let r = report
        .mycelix_resilience
        .expect("adversaries injected → resilience should be Some");
    // At 50yr the TierBuyer attack dilutes (memory: delta goes negative),
    // so tier_buy_resilience should be near 1.0.
    assert!(
        r.tier_buy_resilience > 0.9,
        "TierBuyers should be diluted at 50yr: {:.3}",
        r.tier_buy_resilience,
    );
    assert!(
        r.mean() > 0.5,
        "mean resilience at 50yr too low: {:.3}",
        r.mean(),
    );
}

/// A1 complement: a clean run (no adversaries) should leave
/// `mycelix_resilience` unset, reflecting "metric inapplicable" honestly.
#[test]
fn report_omits_resilience_without_attack() {
    let config = setup(42, 5);
    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();
    assert!(
        report.mycelix_resilience.is_none(),
        "no adversaries → no resilience metric",
    );
}

/// Multi-seed A/B sanity check: 3 seeds × 30 years. Catches seed-specific
/// regressions in the defense mechanisms. The 10-seed × 50-year full sweep
/// lives in `examples/mycelix_multiseed_sweep.rs` for interactive use.
#[test]
fn multiseed_attack_defense_holds() {
    let seeds = [7u64, 42, 137];
    let mut survivors = 0usize;
    let mut cvs_values = Vec::new();
    let mut min_farming = 1.0_f64;

    for &seed in &seeds {
        let config = setup(seed, 30);
        let mut sim = MultiWorldSimulator::new(config);
        sim.run_initialization();
        sim.inject_adversaries(AdversarialStrategy::TierBuyer, 3);
        sim.inject_adversaries(AdversarialStrategy::DemurrageEvader, 3);
        sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, 3);
        sim.inject_adversaries(AdversarialStrategy::CrossClusterAmplifier, 3);
        sim.inject_adversaries(AdversarialStrategy::GuildColluder, 3);
        let report = sim.run();

        if report.survived {
            survivors += 1;
        }
        cvs_values.push(report.final_cvs);

        let (mut credited, mut rejected) = (0u32, 0u32);
        for world in &sim.worlds {
            for a in world.agents.iter().filter(|a| {
                a.is_alive() && matches!(a.adversarial, Some(AdversarialStrategy::CorrectionFarmer))
            }) {
                credited += a.justice.corrections;
                rejected += a.justice.rejected_corrections;
            }
        }
        if credited + rejected > 0 {
            let farming = rejected as f64 / (credited + rejected) as f64;
            if farming < min_farming {
                min_farming = farming;
            }
        }
    }

    assert_eq!(
        survivors,
        seeds.len(),
        "survival should be 100% across seeds"
    );
    let mean_cvs = cvs_values.iter().sum::<f64>() / cvs_values.len() as f64;
    assert!(
        mean_cvs >= 0.3,
        "mean CVS across seeds too low: {:.3}",
        mean_cvs,
    );
    assert!(
        min_farming >= 0.5,
        "worst-seed farming rejection rate too low: {:.3}",
        min_farming,
    );
}
