//! Federated Learning Demo CLI
//!
//! Demonstrates all fl-aggregator capabilities in action:
//! - Byzantine-resistant aggregation
//! - Adaptive defense
//! - Phi tracking
//! - Shapley attribution
//! - Attack simulation
//!
//! Usage:
//!   cargo run --bin fl-demo
//!   cargo run --bin fl-demo -- --honest 10 --byzantine 5 --rounds 20
//!   cargo run --bin fl-demo -- --attack scaling --factor 10

use fl_aggregator::demo::{FLSimulator, SimulationConfig, SimulationReport};
use fl_aggregator::attacks::AttackType;
use std::env;

fn main() {
    println!();
    println!("  ╔═══════════════════════════════════════════════════════════╗");
    println!("  ║     FL-AGGREGATOR: Byzantine-Resistant Federated Learning ║");
    println!("  ║                    End-to-End Demo                        ║");
    println!("  ╚═══════════════════════════════════════════════════════════╝");
    println!();

    let config = parse_args();

    println!("  Configuration:");
    println!("  ─────────────────────────────────────────────────────────────");
    println!("    Honest nodes:     {}", config.honest_nodes);
    println!("    Byzantine nodes:  {}", config.byzantine_nodes);
    println!("    Byzantine ratio:  {:.1}%", config.byzantine_ratio() * 100.0);
    println!("    Rounds:           {}", config.rounds);
    println!("    Gradient dim:     {}", config.gradient_dim);
    println!("    Attack type:      {:?}", config.attack_type);
    println!("    Adaptive defense: {}", if config.use_adaptive_defense { "Yes" } else { "No" });
    println!("    Replay detection: {}", if config.enable_replay_detection { "Yes" } else { "No" });
    println!("  ─────────────────────────────────────────────────────────────");
    println!();

    println!("  Running simulation...\n");

    let mut simulator = FLSimulator::new(config);
    let report = simulator.run();

    println!("\n{}", report);

    // Print ASCII visualization of Phi trend
    println!("  Phi Evolution:");
    println!("  ─────────────────────────────────────────────────────────────");
    print_phi_chart(&report);
    println!();

    // Summary verdict
    print_verdict(&report);
}

fn parse_args() -> SimulationConfig {
    let args: Vec<String> = env::args().collect();
    let mut config = SimulationConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--honest" | "-h" => {
                if i + 1 < args.len() {
                    config.honest_nodes = args[i + 1].parse().unwrap_or(7);
                    i += 1;
                }
            }
            "--byzantine" | "-b" => {
                if i + 1 < args.len() {
                    config.byzantine_nodes = args[i + 1].parse().unwrap_or(3);
                    i += 1;
                }
            }
            "--rounds" | "-r" => {
                if i + 1 < args.len() {
                    config.rounds = args[i + 1].parse().unwrap_or(10);
                    i += 1;
                }
            }
            "--dim" | "-d" => {
                if i + 1 < args.len() {
                    config.gradient_dim = args[i + 1].parse().unwrap_or(1000);
                    i += 1;
                }
            }
            "--attack" | "-a" => {
                if i + 1 < args.len() {
                    config.attack_type = parse_attack(&args[i + 1], &args, i + 2);
                    i += 1;
                }
            }
            "--factor" | "-f" => {
                // Factor for scaling attack (handled in parse_attack)
                i += 1;
            }
            "--no-adaptive" => {
                config.use_adaptive_defense = false;
            }
            "--no-replay" => {
                config.enable_replay_detection = false;
            }
            "--quiet" | "-q" => {
                config.verbose = false;
            }
            "--seed" | "-s" => {
                if i + 1 < args.len() {
                    config.seed = args[i + 1].parse().unwrap_or(42);
                    i += 1;
                }
            }
            "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    config
}

fn parse_attack(name: &str, args: &[String], factor_idx: usize) -> AttackType {
    let factor = if factor_idx < args.len() && (args.get(factor_idx - 1).map(|s| s.as_str()) == Some("--factor") || args.get(factor_idx - 1).map(|s| s.as_str()) == Some("-f")) {
        args.get(factor_idx).and_then(|s| s.parse().ok()).unwrap_or(5.0)
    } else {
        5.0
    };

    match name.to_lowercase().as_str() {
        "scaling" | "scale" => AttackType::GradientScaling { factor },
        "flip" | "signflip" => AttackType::SignFlip,
        "label" | "labelflip" => AttackType::LabelFlip,
        "random" => AttackType::RandomGradient,
        "zero" => AttackType::ZeroGradient,
        "noise" => AttackType::GradientNoise { std_dev: factor },
        "adaptive" => AttackType::AdaptiveAttack,
        "freerider" | "free" => AttackType::FreeRider,
        _ => AttackType::GradientScaling { factor: 5.0 },
    }
}

fn print_help() {
    println!("FL-Aggregator Demo - Byzantine-Resistant Federated Learning");
    println!();
    println!("USAGE:");
    println!("  fl-demo [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --honest, -h <N>      Number of honest nodes (default: 7)");
    println!("  --byzantine, -b <N>   Number of Byzantine nodes (default: 3)");
    println!("  --rounds, -r <N>      Number of training rounds (default: 10)");
    println!("  --dim, -d <N>         Gradient dimension (default: 1000)");
    println!("  --attack, -a <TYPE>   Attack type: scaling, flip, label, random, zero,");
    println!("                        noise, adaptive, freerider (default: scaling)");
    println!("  --factor, -f <N>      Attack factor for scaling/noise (default: 5.0)");
    println!("  --no-adaptive         Disable adaptive defense");
    println!("  --no-replay           Disable replay detection");
    println!("  --quiet, -q           Reduce output verbosity");
    println!("  --seed, -s <N>        Random seed (default: 42)");
    println!("  --help                Show this help message");
    println!();
    println!("EXAMPLES:");
    println!("  fl-demo                           # Run with defaults");
    println!("  fl-demo --byzantine 5 --rounds 20 # More attackers, longer run");
    println!("  fl-demo --attack adaptive         # Test adaptive attack");
    println!("  fl-demo --attack scaling -f 10    # 10x gradient scaling attack");
}

fn print_phi_chart(report: &SimulationReport) {
    let phi_values: Vec<f32> = report.rounds.iter().map(|r| r.phi_value).collect();

    if phi_values.is_empty() {
        return;
    }

    let max_phi = phi_values.iter().cloned().fold(0.0f32, f32::max);
    let min_phi = phi_values.iter().cloned().fold(1.0f32, f32::min);
    let range = (max_phi - min_phi).max(0.1);

    let height = 8;
    let width = phi_values.len().min(50);

    for row in (0..height).rev() {
        let threshold = min_phi + (row as f32 / height as f32) * range;
        print!("    {:>5.2} │", threshold);

        for (i, &phi) in phi_values.iter().enumerate().take(width) {
            if phi >= threshold {
                // Color based on trend
                let trend = &report.rounds[i].phi_trend;
                let symbol = match trend.as_str() {
                    "Rising" => "▲",
                    "Falling" => "▼",
                    "Volatile" => "◆",
                    _ => "●",
                };
                print!(" {}", symbol);
            } else {
                print!("  ");
            }
        }
        println!();
    }

    print!("          └");
    for _ in 0..width {
        print!("──");
    }
    println!();
    print!("           ");
    for i in 0..width {
        if i % 5 == 0 {
            print!("{:<2}", i + 1);
        } else {
            print!("  ");
        }
    }
    println!(" Round");
}

fn print_verdict(report: &SimulationReport) {
    println!("  ═══════════════════════════════════════════════════════════════");
    println!("  VERDICT");
    println!("  ═══════════════════════════════════════════════════════════════");

    let detection_rate = if report.config.byzantine_nodes > 0 {
        report.total_byzantine_detected as f32
            / (report.config.byzantine_nodes * report.config.rounds) as f32
    } else {
        1.0
    };

    let quality = report.average_aggregation_quality;
    let phi_healthy = report.average_phi > 0.3;

    let overall = if quality > 0.9 && detection_rate > 0.5 && phi_healthy {
        "EXCELLENT - System maintained high integrity"
    } else if quality > 0.7 && (detection_rate > 0.3 || phi_healthy) {
        "GOOD - System resisted most attacks"
    } else if quality > 0.5 {
        "FAIR - Some degradation but still functional"
    } else {
        "POOR - Attack significantly impacted aggregation"
    };

    println!();
    println!("    Detection Rate:       {:.1}%", detection_rate * 100.0);
    println!("    Aggregation Quality:  {:.1}%", quality * 100.0);
    println!("    System Coherence:     {} (Phi={:.3})",
        if phi_healthy { "Healthy" } else { "Degraded" },
        report.average_phi
    );
    println!();
    println!("    Overall: {}", overall);
    println!();
    println!("  ═══════════════════════════════════════════════════════════════");
    println!();
}
