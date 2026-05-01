// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ethical Pluralism Experiment: Does consciousness-gated governance improve
//! outcomes regardless of the population's ethical composition?
//!
//! 7 conditions × N seeds:
//!   A: Homogeneous default ethics + equal-weight governance
//!   B: Homogeneous default ethics + consciousness-gated governance
//!   C: Pluralistic (random) ethics + equal-weight governance
//!   D: Pluralistic (random) ethics + consciousness-gated governance
//!   E: 80% deontological + consciousness-gated governance
//!   F: 80% consequentialist + consciousness-gated governance
//!   G: 80% relational + consciousness-gated governance
//!
//! Key hypothesis: If D > C AND B > A, consciousness-gating is robust to
//! ethical diversity — a much stronger claim than "one ethical framework benefits."
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --bin ethical_pluralism_experiment
//! cargo run --release --bin ethical_pluralism_experiment -- --seeds 50 --ticks 1800
//! ```

use mycelix_multiworld_sim::agent::EthicalOrientation;
use mycelix_multiworld_sim::config::{
    EpochConfig, PolicyConfig, SimulationConfig, WorldSeedConfig,
};
use mycelix_multiworld_sim::statistics::{bootstrap_ci, cohens_d, paired_t_test};
use mycelix_multiworld_sim::stochastic::StochasticEngine;
use mycelix_multiworld_sim::MultiWorldSimulator;

/// How to set founding agents' ethical orientations.
#[derive(Clone)]
enum EthicsProfile {
    /// Default: all agents get EthicalOrientation::default() (0.4 uniform).
    Homogeneous,
    /// Random: each agent drawn from Uniform(0.1, 0.9) per dimension.
    Pluralistic,
    /// Dominated: 80% of agents get the dominant orientation, 20% random.
    Dominated { dominant: [f64; 4] },
}

/// Strongly deontological: high duty, low outcome-focus.
const DEONTOLOGICAL: [f64; 4] = [0.85, 0.25, 0.35, 0.30];
/// Strongly consequentialist: high outcome-focus, low duty.
const CONSEQUENTIALIST: [f64; 4] = [0.25, 0.85, 0.30, 0.35];
/// Strongly relational/Ubuntu: high community, low individualism.
const RELATIONAL: [f64; 4] = [0.30, 0.30, 0.40, 0.85];

struct Condition {
    name: &'static str,
    trust_weighted: bool,
    ethics_profile: EthicsProfile,
}

fn conditions() -> Vec<Condition> {
    vec![
        Condition {
            name: "A: homo+equal",
            trust_weighted: false,
            ethics_profile: EthicsProfile::Homogeneous,
        },
        Condition {
            name: "B: homo+gated",
            trust_weighted: true,
            ethics_profile: EthicsProfile::Homogeneous,
        },
        Condition {
            name: "C: plural+equal",
            trust_weighted: false,
            ethics_profile: EthicsProfile::Pluralistic,
        },
        Condition {
            name: "D: plural+gated",
            trust_weighted: true,
            ethics_profile: EthicsProfile::Pluralistic,
        },
        Condition {
            name: "E: deont+gated",
            trust_weighted: true,
            ethics_profile: EthicsProfile::Dominated {
                dominant: DEONTOLOGICAL,
            },
        },
        Condition {
            name: "F: conseq+gated",
            trust_weighted: true,
            ethics_profile: EthicsProfile::Dominated {
                dominant: CONSEQUENTIALIST,
            },
        },
        Condition {
            name: "G: relat+gated",
            trust_weighted: true,
            ethics_profile: EthicsProfile::Dominated {
                dominant: RELATIONAL,
            },
        },
    ]
}

fn make_config(seed: u64, ticks: u32, trust_weighted: bool) -> SimulationConfig {
    let mut policy = PolicyConfig::default();
    policy.trust_weighted_governance = trust_weighted;

    SimulationConfig {
        total_ticks: ticks,
        seed,
        initial_worlds: vec![
            WorldSeedConfig {
                name: "Earth".into(),
                location: "Earth".into(),
                founding_tick: 0,
                initial_population: 500,
                initial_resources: 1.0,
            },
            WorldSeedConfig {
                name: "Artemis Base".into(),
                location: "Moon".into(),
                founding_tick: 0,
                initial_population: 30,
                initial_resources: 0.3,
            },
        ],
        epoch_configs: vec![EpochConfig {
            id: 0,
            name: "EthicalPluralism".into(),
            start_tick: 0,
            end_tick: ticks,
            population_trigger: None,
            self_sufficiency_trigger: None,
        }],
        policy,
    }
}

/// Override founding agents' ethics after construction but before run().
///
/// CRITICAL: Uses sim.rng (not a separate RNG) so that different ethical
/// profiles produce different RNG states, causing the simulation to
/// genuinely diverge. With a separate RNG, the sim's main RNG stays
/// identical across conditions, making all Bernoulli outcomes the same
/// regardless of ethics — the first experiment's false-negative.
fn apply_ethics(sim: &mut MultiWorldSimulator, profile: &EthicsProfile) {
    for wi in 0..sim.worlds.len() {
        let n = sim.worlds[wi].agents.len().max(1);
        for i in 0..sim.worlds[wi].agents.len() {
            if !sim.worlds[wi].agents[i].is_alive() {
                continue;
            }
            match profile {
                EthicsProfile::Homogeneous => {
                    sim.worlds[wi].agents[i].ethics = EthicalOrientation::default();
                    // NO dummy RNG calls — Homogeneous leaves RNG state unchanged.
                    // Other profiles advance RNG, causing trajectory divergence.
                }
                EthicsProfile::Pluralistic => {
                    sim.worlds[wi].agents[i].ethics = EthicalOrientation {
                        deontological: 0.1 + sim.rng.next_f64() * 0.8,
                        consequentialist: 0.1 + sim.rng.next_f64() * 0.8,
                        virtue_care: 0.1 + sim.rng.next_f64() * 0.8,
                        relational: 0.1 + sim.rng.next_f64() * 0.8,
                    };
                }
                EthicsProfile::Dominated { dominant } => {
                    if (i as f64 / n as f64) < 0.8 {
                        sim.worlds[wi].agents[i].ethics = EthicalOrientation {
                            deontological: (dominant[0] + sim.rng.next_gaussian(0.0, 0.05))
                                .clamp(0.05, 1.0),
                            consequentialist: (dominant[1] + sim.rng.next_gaussian(0.0, 0.05))
                                .clamp(0.05, 1.0),
                            virtue_care: (dominant[2] + sim.rng.next_gaussian(0.0, 0.05))
                                .clamp(0.05, 1.0),
                            relational: (dominant[3] + sim.rng.next_gaussian(0.0, 0.05))
                                .clamp(0.05, 1.0),
                        };
                    } else {
                        sim.worlds[wi].agents[i].ethics = EthicalOrientation {
                            deontological: 0.1 + sim.rng.next_f64() * 0.8,
                            consequentialist: 0.1 + sim.rng.next_f64() * 0.8,
                            virtue_care: 0.1 + sim.rng.next_f64() * 0.8,
                            relational: 0.1 + sim.rng.next_f64() * 0.8,
                        };
                    }
                }
            }
        }
    }
}

struct RunResult {
    cvs: f64,
    phi: f64,
    population: usize,
}

fn run_condition(seed: u64, ticks: u32, cond: &Condition) -> RunResult {
    let mut sim = MultiWorldSimulator::new(make_config(seed, ticks, cond.trust_weighted));
    // Force world initialization (normally happens at start of run()).
    // We need agents to exist before apply_ethics modifies them.
    // run() will skip re-initialization because worlds is no longer empty.
    sim.run_initialization();
    apply_ethics(&mut sim, &cond.ethics_profile);
    let report = sim.run();
    let phi = sim
        .worlds
        .iter()
        .map(|w| w.mean_phi() * w.population() as f64)
        .sum::<f64>()
        / report.final_population.max(1) as f64;
    RunResult {
        cvs: report.final_cvs,
        phi,
        population: report.final_population,
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_seeds: usize = args
        .iter()
        .position(|a| a == "--seeds")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let ticks: u32 = args
        .iter()
        .position(|a| a == "--ticks")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1800);

    let seeds: Vec<u64> = (0..n_seeds).map(|i| 42 + i as u64 * 17).collect();
    let conds = conditions();

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  ETHICAL PLURALISM EXPERIMENT                               ║");
    eprintln!(
        "║  {} conditions × {} seeds × {} ticks ({:.0} years)           ║",
        conds.len(),
        n_seeds,
        ticks,
        ticks as f64 / 12.0
    );
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut all_cvs: Vec<Vec<f64>> = vec![Vec::new(); conds.len()];
    let mut all_phi: Vec<Vec<f64>> = vec![Vec::new(); conds.len()];
    let mut all_pop: Vec<Vec<f64>> = vec![Vec::new(); conds.len()];

    for (ci, cond) in conds.iter().enumerate() {
        eprint!("  {} ", cond.name);
        for &seed in &seeds {
            eprint!(".");
            let result = run_condition(seed, ticks, cond);
            all_cvs[ci].push(result.cvs);
            all_phi[ci].push(result.phi);
            all_pop[ci].push(result.population as f64);
        }
        eprintln!(
            " CVS={:.4} Phi={:.4} Pop={:.0}",
            mean(&all_cvs[ci]),
            mean(&all_phi[ci]),
            mean(&all_pop[ci])
        );
    }

    let mut rng = StochasticEngine::new(12345);

    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  RESULTS                                                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    println!(
        "{:<20} {:>8} {:>14} {:>8} {:>8}",
        "Condition", "CVS", "CVS 95% CI", "Phi", "Pop"
    );
    println!("{}", "-".repeat(62));

    for (ci, cond) in conds.iter().enumerate() {
        let ci_result = bootstrap_ci(&all_cvs[ci], 0.95, 2000, &mut rng);
        println!(
            "{:<20} {:>8.4} [{:>5.4}, {:>5.4}] {:>8.4} {:>8.0}",
            cond.name,
            mean(&all_cvs[ci]),
            ci_result.as_ref().map(|c| c.lower).unwrap_or(0.0),
            ci_result.as_ref().map(|c| c.upper).unwrap_or(0.0),
            mean(&all_phi[ci]),
            mean(&all_pop[ci]),
        );
    }

    println!("\n{}", "=".repeat(62));
    println!("PAIRWISE COMPARISONS (Cohen's d, paired t-test p-value)\n");

    let comparisons: &[(&str, usize, usize)] = &[
        ("B vs A (gating effect, homo)", 1, 0),
        ("D vs C (gating effect, plural)", 3, 2),
        ("C vs A (pluralism, ungated)", 2, 0),
        ("D vs B (pluralism, gated)", 3, 1),
        ("E vs D (deont vs plural)", 4, 3),
        ("F vs D (conseq vs plural)", 5, 3),
        ("G vs D (relat vs plural)", 6, 3),
        ("E vs F (deont vs conseq)", 4, 5),
        ("E vs G (deont vs relat)", 4, 6),
        ("F vs G (conseq vs relat)", 5, 6),
    ];

    println!(
        "{:<35} {:>8} {:>8} {:>8} {:>10}",
        "Comparison", "ΔCVS", "d", "p", "Sig?"
    );
    println!("{}", "-".repeat(73));

    for &(label, better_idx, worse_idx) in comparisons {
        let delta = mean(&all_cvs[better_idx]) - mean(&all_cvs[worse_idx]);
        let d = cohens_d(&all_cvs[better_idx], &all_cvs[worse_idx]).unwrap_or(0.0);
        let p = paired_t_test(&all_cvs[better_idx], &all_cvs[worse_idx])
            .map(|t| t.p_value)
            .unwrap_or(1.0);
        let sig = if p < 0.001 {
            "***"
        } else if p < 0.01 {
            "**"
        } else if p < 0.05 {
            "*"
        } else {
            "ns"
        };

        println!(
            "{:<35} {:>+8.4} {:>8.3} {:>8.4} {:>10}",
            label, delta, d, p, sig
        );
    }

    println!("\n{}", "=".repeat(62));
    println!("VERDICTS\n");

    let gating_homo = mean(&all_cvs[1]) - mean(&all_cvs[0]);
    let gating_plural = mean(&all_cvs[3]) - mean(&all_cvs[2]);
    let p_homo = paired_t_test(&all_cvs[1], &all_cvs[0])
        .map(|t| t.p_value)
        .unwrap_or(1.0);
    let p_plural = paired_t_test(&all_cvs[3], &all_cvs[2])
        .map(|t| t.p_value)
        .unwrap_or(1.0);

    println!("1. CONSCIOUSNESS-GATING ROBUSTNESS:");
    if gating_homo > 0.0 && gating_plural > 0.0 && p_homo < 0.05 && p_plural < 0.05 {
        println!("   CONFIRMED: Consciousness-gated governance improves CVS");
        println!(
            "   REGARDLESS of ethical composition (homo: +{:.4}, plural: +{:.4}).",
            gating_homo, gating_plural
        );
    } else if gating_homo > 0.0 || gating_plural > 0.0 {
        println!("   PARTIAL: Gating helps in some conditions but not all.");
        println!(
            "   Homo: {:+.4} (p={:.4}), Plural: {:+.4} (p={:.4}).",
            gating_homo, p_homo, gating_plural, p_plural
        );
    } else {
        println!("   NOT CONFIRMED: Gating does not reliably improve outcomes.");
    }

    let plural_effect = mean(&all_cvs[3]) - mean(&all_cvs[1]);
    println!("\n2. ETHICAL DIVERSITY EFFECT:");
    if plural_effect > 0.01 {
        println!(
            "   Pluralistic ethics OUTPERFORM homogeneous by {:.4} CVS.",
            plural_effect
        );
    } else if plural_effect < -0.01 {
        println!(
            "   Homogeneous ethics outperform pluralistic by {:.4} CVS.",
            -plural_effect
        );
    } else {
        println!("   No significant difference between pluralistic and homogeneous ethics.");
    }

    let best_gated = (4..7)
        .max_by(|&a, &b| {
            mean(&all_cvs[a])
                .partial_cmp(&mean(&all_cvs[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(4);
    let best_name = conds[best_gated].name;
    println!("\n3. OPTIMAL ETHICAL COMPOSITION (under gating):");
    println!(
        "   Best: {} (CVS={:.4})",
        best_name,
        mean(&all_cvs[best_gated])
    );
    println!(
        "   vs Pluralistic D: {:+.4} CVS",
        mean(&all_cvs[best_gated]) - mean(&all_cvs[3])
    );

    println!("\n--- CSV ---");
    println!("condition,seed,cvs,phi,population");
    for (ci, cond) in conds.iter().enumerate() {
        for (si, &seed) in seeds.iter().enumerate() {
            println!(
                "{},{},{:.6},{:.6},{:.0}",
                cond.name, seed, all_cvs[ci][si], all_phi[ci][si], all_pop[ci][si]
            );
        }
    }
}
