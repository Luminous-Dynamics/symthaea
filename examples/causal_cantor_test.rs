//! # Causal Cantor Network Benchmark
//!
//! Tests the novel Hierarchical Cantor-LTC approach to causal discovery
//! on the Tübingen benchmark dataset.
//!
//! This is a fundamentally different approach from flat ensemble methods:
//! - Uses multi-timescale hierarchical integration
//! - Leverages the Cantor-LTC architecture for evidence aggregation
//! - Uses Φ (integrated information) for confidence estimation

use symthaea::benchmarks::{
    TuebingenAdapter,
    CausalDirection,
    CausalCantorNetwork,
    discover_by_cantor,
    train_causal_cantor,
    // Comparison methods
    discover_majority_voting,
    discover_information_theoretic,
    ReciDiscovery,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║           CAUSAL CANTOR NETWORK - HIERARCHICAL DISCOVERY                 ║");
    println!("║                    Multi-Timescale Evidence Integration                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Load Tübingen benchmark
    let tuebingen_path = "benchmarks/external/tuebingen";
    let adapter = match TuebingenAdapter::load(tuebingen_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to load dataset: {}", e);
            return;
        }
    };
    println!("Loaded {} cause-effect pairs\n", adapter.len());

    // =========================================================================
    // TEST 1: Untrained CausalCantorNetwork
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 1: UNTRAINED CAUSAL CANTOR NETWORK                                 │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    let results_untrained = adapter.run(|x, y| discover_by_cantor(x, y).direction);
    println!("  Untrained CausalCantor...");
    println!("         Accuracy: {:.1}% ({}/{})",
             results_untrained.accuracy() * 100.0,
             results_untrained.correct,
             results_untrained.total);

    // =========================================================================
    // TEST 2: Trained CausalCantorNetwork (Leave-One-Out CV)
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 2: TRAINED CAUSAL CANTOR NETWORK (LEAVE-ONE-OUT CV)                │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    let pairs = adapter.get_pairs();

    // Prepare training data
    let training_data: Vec<(Vec<f64>, Vec<f64>, CausalDirection)> = pairs.iter()
        .map(|p| (p.x.clone(), p.y.clone(), p.ground_truth.clone()))
        .collect();

    // Leave-one-out cross-validation
    let mut correct = 0;
    let mut total = 0;
    let mut phi_sum = 0.0;

    println!("  Running leave-one-out CV...");
    for (i, pair) in pairs.iter().enumerate() {
        // Train on all pairs except this one
        let cv_training: Vec<_> = training_data.iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, p)| p.clone())
            .collect();

        let mut network = CausalCantorNetwork::new();
        network.train(&cv_training);

        // Test on held-out pair
        let result = network.discover(&pair.x, &pair.y);
        phi_sum += network.get_phi();

        let is_correct = match (&result.direction, &pair.ground_truth) {
            (CausalDirection::Forward, CausalDirection::Forward) => true,
            (CausalDirection::Backward, CausalDirection::Backward) => true,
            _ => false,
        };

        if is_correct {
            correct += 1;
        }
        total += 1;

        // Progress indicator
        if (i + 1) % 20 == 0 {
            println!("    Processed {}/{} pairs...", i + 1, pairs.len());
        }
    }

    let cv_accuracy = correct as f64 / total as f64;
    let avg_phi = phi_sum / total as f64;

    println!("\n  Trained CausalCantor (LOO-CV)...");
    println!("         Accuracy: {:.1}% ({}/{})", cv_accuracy * 100.0, correct, total);
    println!("         Average Φ: {:.4}", avg_phi);

    // =========================================================================
    // TEST 3: Compare with Different Network Configurations
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 3: NETWORK CONFIGURATION COMPARISON                                │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    // Different depth configurations
    for depth in [3, 5, 7] {
        let mut correct = 0;
        let total = pairs.len();

        for (i, pair) in pairs.iter().enumerate() {
            let cv_training: Vec<_> = training_data.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, p)| p.clone())
                .collect();

            let mut network = CausalCantorNetwork::with_config(depth, 100.0, 50);
            network.train(&cv_training);

            let result = network.discover(&pair.x, &pair.y);

            let is_correct = match (&result.direction, &pair.ground_truth) {
                (CausalDirection::Forward, CausalDirection::Forward) => true,
                (CausalDirection::Backward, CausalDirection::Backward) => true,
                _ => false,
            };

            if is_correct {
                correct += 1;
            }
        }

        let accuracy = correct as f64 / total as f64;
        println!("  Depth={}: Accuracy = {:.1}% ({}/{})", depth, accuracy * 100.0, correct, total);
    }

    // =========================================================================
    // COMPARISON WITH BASELINE METHODS
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ COMPARISON WITH BASELINE METHODS                                        │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    // RECI (best single method)
    let reci = ReciDiscovery::new();
    let results_reci = adapter.run(|x, y| reci.discover(x, y).direction);
    println!("  RECI (best primitive)...");
    println!("         Accuracy: {:.1}% ({}/{})",
             results_reci.accuracy() * 100.0,
             results_reci.correct,
             results_reci.total);

    // Info-Theoretic
    let results_info = adapter.run(discover_information_theoretic);
    println!("  Info-Theoretic...");
    println!("         Accuracy: {:.1}% ({}/{})",
             results_info.accuracy() * 100.0,
             results_info.correct,
             results_info.total);

    // Majority Voting (current best)
    let results_majority = adapter.run(discover_majority_voting);
    println!("  Majority Voting (previous best)...");
    println!("         Accuracy: {:.1}% ({}/{})",
             results_majority.accuracy() * 100.0,
             results_majority.correct,
             results_majority.total);

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                            FINAL SUMMARY                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let methods = vec![
        ("Untrained CausalCantor", results_untrained.accuracy()),
        ("Trained CausalCantor (CV)", cv_accuracy),
        ("RECI", results_reci.accuracy()),
        ("Info-Theoretic", results_info.accuracy()),
        ("Majority Voting", results_majority.accuracy()),
    ];

    println!("  Method                        Accuracy    vs Majority");
    println!("  ────────────────────────────────────────────────────────");

    let majority_acc = results_majority.accuracy();
    for (name, acc) in &methods {
        let delta = (acc - majority_acc) * 100.0;
        let marker = if delta > 0.0 { "↑" } else if delta < 0.0 { "↓" } else { "=" };
        println!("  {:28} {:5.1}%      {:+5.1}% {}", name, acc * 100.0, delta, marker);
    }

    // Best method
    let best = methods.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!("\n  ════════════════════════════════════════════════════════");
    println!("  BEST METHOD: {} ({:.1}%)", best.0, best.1 * 100.0);
    println!("  GAP TO 100%: {:.1}%", (1.0 - best.1) * 100.0);

    if cv_accuracy > majority_acc {
        println!("\n  ✓ CausalCantor EXCEEDS Majority Voting by {:.1}%!",
                 (cv_accuracy - majority_acc) * 100.0);
    } else if cv_accuracy == majority_acc {
        println!("\n  ~ CausalCantor matches Majority Voting");
    } else {
        println!("\n  Gap: {:.1}% (CausalCantor: {:.1}% vs Majority: {:.1}%)",
                 (majority_acc - cv_accuracy) * 100.0,
                 cv_accuracy * 100.0, majority_acc * 100.0);
    }

    // =========================================================================
    // Φ ANALYSIS
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ Φ (INTEGRATED INFORMATION) ANALYSIS                                     │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    // Analyze Φ distribution for correct vs incorrect predictions
    let mut phi_correct = Vec::new();
    let mut phi_incorrect = Vec::new();

    for (i, pair) in pairs.iter().enumerate() {
        let cv_training: Vec<_> = training_data.iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, p)| p.clone())
            .collect();

        let mut network = CausalCantorNetwork::new();
        network.train(&cv_training);

        let result = network.discover(&pair.x, &pair.y);
        let phi = network.get_phi();

        let is_correct = match (&result.direction, &pair.ground_truth) {
            (CausalDirection::Forward, CausalDirection::Forward) => true,
            (CausalDirection::Backward, CausalDirection::Backward) => true,
            _ => false,
        };

        if is_correct {
            phi_correct.push(phi);
        } else {
            phi_incorrect.push(phi);
        }
    }

    let avg_phi_correct = if !phi_correct.is_empty() {
        phi_correct.iter().sum::<f64>() / phi_correct.len() as f64
    } else {
        0.0
    };

    let avg_phi_incorrect = if !phi_incorrect.is_empty() {
        phi_incorrect.iter().sum::<f64>() / phi_incorrect.len() as f64
    } else {
        0.0
    };

    println!("  Correct predictions:   Φ = {:.4} (n={})", avg_phi_correct, phi_correct.len());
    println!("  Incorrect predictions: Φ = {:.4} (n={})", avg_phi_incorrect, phi_incorrect.len());

    if avg_phi_correct > avg_phi_incorrect {
        println!("\n  ✓ Higher Φ correlates with correct predictions!");
        println!("    This suggests Φ could be used for confidence-based abstention.");
    } else {
        println!("\n  ~ Φ does not clearly separate correct/incorrect predictions.");
    }

    println!("\n  Done!");
}
