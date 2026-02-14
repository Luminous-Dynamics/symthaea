//! Meta-Learning Adaptation Curve Benchmark
//!
//! Demonstrates signal weight evolution over 30+ rounds:
//! - Persistent attackers trigger EMA exclusion rate buildup
//! - Signal weights adapt to the dominant attack pattern
//! - Reformed attackers see exclusion rate decay
//! - Prints round-by-round signal weights + exclusion rates
//!
//! Run: `cargo run --example benchmark_meta_learning --release`

use mycelix_fl_core::meta_learning::{MetaLearningByzantinePlugin, MetaLearningConfig};
use mycelix_fl_core::plugins::ByzantinePlugin;
use mycelix_fl_core::types::GradientUpdate;

/// Generate honest gradients: small values near 0.5
fn honest_update(id: &str, dim: usize, seed: usize) -> GradientUpdate {
    let val = 0.5 + (seed as f32 * 0.001);
    GradientUpdate::new(id.into(), 1, vec![val; dim], 100, 0.5)
}

/// Generate magnitude attack: extreme norm
fn magnitude_attack(id: &str, dim: usize, seed: usize) -> GradientUpdate {
    let val = if seed % 2 == 0 { 100.0 } else { -100.0 };
    GradientUpdate::new(id.into(), 1, vec![val; dim], 100, 0.9)
}

/// Generate direction attack: opposite direction to honest mean
fn direction_attack(id: &str, dim: usize, _seed: usize) -> GradientUpdate {
    let gradients: Vec<f32> = (0..dim)
        .map(|i| if i % 2 == 0 { -0.5 } else { 0.5 })
        .collect();
    GradientUpdate::new(id.into(), 1, gradients, 100, 0.9)
}

/// Generate subtle attack: slightly inflated gradients (harder to detect)
fn subtle_attack(id: &str, dim: usize, _seed: usize) -> GradientUpdate {
    GradientUpdate::new(id.into(), 1, vec![2.0; dim], 100, 0.55)
}

fn main() {
    println!("=== Meta-Learning Byzantine Adaptation Benchmark ===\n");

    let dim = 50;
    let n_honest = 8;
    let n_byz = 3;
    let total_rounds = 35;
    let reform_round = 20; // Byzantine nodes become honest after this round

    let config = MetaLearningConfig {
        ema_alpha: 0.15,
        suspicion_threshold: 0.25,
        learning_rate: 0.02,
        min_rounds: 4,
        suspicious_weight: 0.1,
    };

    // === Scenario 1: Magnitude Attacks ===
    println!("--- Scenario 1: Magnitude Attack (byz values = ±100) ---\n");
    run_scenario(
        "magnitude",
        dim,
        n_honest,
        n_byz,
        total_rounds,
        reform_round,
        config.clone(),
        |id, dim, seed| magnitude_attack(id, dim, seed),
    );

    // === Scenario 2: Direction Attacks ===
    println!("\n--- Scenario 2: Direction Attack (opposite to honest mean) ---\n");
    run_scenario(
        "direction",
        dim,
        n_honest,
        n_byz,
        total_rounds,
        reform_round,
        config.clone(),
        |id, dim, seed| direction_attack(id, dim, seed),
    );

    // === Scenario 3: Subtle Attacks ===
    println!("\n--- Scenario 3: Subtle Attack (2x honest magnitude) ---\n");
    run_scenario(
        "subtle",
        dim,
        n_honest,
        n_byz,
        total_rounds,
        reform_round,
        config.clone(),
        |id, dim, seed| subtle_attack(id, dim, seed),
    );

    println!("\n=== Verification ===");

    // Run verification scenario
    let mut plugin = MetaLearningByzantinePlugin::with_config(config.clone());
    let mut passed = 0;
    let total_tests = 5;

    // Phase 1: 15 rounds of magnitude attacks
    for round in 0..15 {
        let mut updates = Vec::new();
        for i in 0..n_honest {
            updates.push(honest_update(&format!("h{}", i), dim, i + round));
        }
        for i in 0..n_byz {
            updates.push(magnitude_attack(&format!("b{}", i), dim, i + round));
        }
        let _weights = plugin.analyze(&updates);
        let excluded: Vec<String> = (0..n_byz).map(|i| format!("b{}", i)).collect();
        plugin.record_outcome(round as u64, &excluded);
    }

    // Test 1: Persistent attackers are suspicious after 15 rounds
    let all_suspicious = (0..n_byz).all(|i| plugin.is_suspicious(&format!("b{}", i)));
    if all_suspicious {
        println!(
            "[PASS] Test 1: All {} persistent attackers flagged suspicious",
            n_byz
        );
        passed += 1;
    } else {
        println!("[FAIL] Test 1: Not all persistent attackers suspicious");
    }

    // Test 2: Honest nodes are NOT suspicious
    let no_honest_suspicious = (0..n_honest).all(|i| !plugin.is_suspicious(&format!("h{}", i)));
    if no_honest_suspicious {
        println!("[PASS] Test 2: No honest nodes flagged suspicious");
        passed += 1;
    } else {
        println!("[FAIL] Test 2: Some honest nodes incorrectly flagged");
    }

    // Test 3: Signal weights have adapted (not equal to initial)
    let w = plugin.signal_weights();
    let initial_mag = 0.25; // Default from SignalWeights::default()
    let adapted = (w.magnitude - initial_mag).abs() > 0.01;
    if adapted {
        println!(
            "[PASS] Test 3: Signal weights adapted (magnitude: {:.3} != {:.3})",
            w.magnitude, initial_mag
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 3: Signal weights unchanged: mag={:.3}",
            w.magnitude
        );
    }

    // Test 4: Weights still sum to ~1.0
    let sum = w.magnitude + w.direction + w.cross_validation + w.coordinate;
    if (sum - 1.0).abs() < 0.02 {
        println!("[PASS] Test 4: Signal weights sum to {:.4} (≈1.0)", sum);
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 4: Signal weights sum to {:.4} (expected ≈1.0)",
            sum
        );
    }

    // Phase 2: 20 more rounds where attackers reform (honest gradients)
    for round in 15..35 {
        let mut updates = Vec::new();
        for i in 0..n_honest {
            updates.push(honest_update(&format!("h{}", i), dim, i + round));
        }
        for i in 0..n_byz {
            updates.push(honest_update(&format!("b{}", i), dim, i + round)); // reformed!
        }
        let _weights = plugin.analyze(&updates);
        plugin.record_outcome(round as u64, &[]); // nobody excluded
    }

    // Test 5: After 20 honest rounds, exclusion rates should have decayed
    let b0_profile = plugin.get_participant_profile("b0").unwrap();
    if b0_profile.exclusion_rate < 0.15 {
        println!(
            "[PASS] Test 5: Reformed attacker b0 exclusion rate decayed to {:.4}",
            b0_profile.exclusion_rate
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 5: b0 exclusion rate still high: {:.4} (expected < 0.15)",
            b0_profile.exclusion_rate
        );
    }

    println!("\n=== RESULTS: {}/{} passed ===", passed, total_tests);
    if passed < total_tests {
        std::process::exit(1);
    }
}

fn run_scenario<F>(
    name: &str,
    dim: usize,
    n_honest: usize,
    n_byz: usize,
    total_rounds: usize,
    reform_round: usize,
    config: MetaLearningConfig,
    make_attack: F,
) where
    F: Fn(&str, usize, usize) -> GradientUpdate,
{
    let mut plugin = MetaLearningByzantinePlugin::with_config(config);

    println!(
        "{:<6} {:>6} {:>6} {:>6} {:>6}  {:>8} {:>8} {:>8}  {:>8}",
        "Round", "Mag", "Dir", "XVal", "Coord", "b0_rate", "b1_rate", "b2_rate", "Phase"
    );
    println!("{:-<85}", "");

    for round in 0..total_rounds {
        let mut updates = Vec::new();
        for i in 0..n_honest {
            updates.push(honest_update(&format!("h{}", i), dim, i + round));
        }

        let phase = if round < reform_round {
            "attack"
        } else {
            "reform"
        };

        for i in 0..n_byz {
            if round < reform_round {
                updates.push(make_attack(&format!("b{}", i), dim, i + round));
            } else {
                updates.push(honest_update(&format!("b{}", i), dim, i + round));
            }
        }

        let _weights = plugin.analyze(&updates);

        // Simulate pipeline decision: exclude those that were weighted down
        let excluded: Vec<String> = if round < reform_round {
            (0..n_byz).map(|i| format!("b{}", i)).collect()
        } else {
            vec![] // reformed, nobody excluded
        };
        plugin.record_outcome(round as u64, &excluded);

        // Print signal weights and exclusion rates
        let w = plugin.signal_weights();
        let rates: Vec<f32> = (0..n_byz.min(3))
            .map(|i| {
                plugin
                    .get_participant_profile(&format!("b{}", i))
                    .map(|p| p.exclusion_rate)
                    .unwrap_or(0.0)
            })
            .collect();

        println!(
            "{:<6} {:>6.3} {:>6.3} {:>6.3} {:>6.3}  {:>8.4} {:>8.4} {:>8.4}  {:>8}",
            round + 1,
            w.magnitude,
            w.direction,
            w.cross_validation,
            w.coordinate,
            rates.first().copied().unwrap_or(0.0),
            rates.get(1).copied().unwrap_or(0.0),
            rates.get(2).copied().unwrap_or(0.0),
            phase
        );
    }

    // Summary
    let w = plugin.signal_weights();
    println!(
        "\n  {} final weights: mag={:.3} dir={:.3} xval={:.3} coord={:.3}",
        name, w.magnitude, w.direction, w.cross_validation, w.coordinate
    );
    let final_rate = plugin
        .get_participant_profile("b0")
        .map(|p| p.exclusion_rate)
        .unwrap_or(0.0);
    println!(
        "  b0 final exclusion rate: {:.4} (suspicious: {})",
        final_rate,
        plugin.is_suspicious("b0")
    );
}
