// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # LEAF Federated Learning Benchmark
//!
//! Tests Symthaea's federated CfC aggregation protocol against
//! the LEAF benchmark framework (Federated EMNIST split).
//! Since LEAF requires specialized data partitioning, this benchmark
//! uses synthetic non-IID client data to validate:
//!
//! 1. FedAvg convergence with trust-weighted aggregation
//! 2. Differential privacy impact on model utility
//! 3. Byzantine fault tolerance (trimmed mean)
//! 4. Gradient staleness handling
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_federated_leaf --release
//! ```

use std::time::Instant;

use ndarray::Array1;
use symthaea::dynamics::cfc::{ActivationType, CfCConfig, CfCNetwork, CfCNetworkConfig};
use symthaea::swarm::{DifferentialPrivacyConfig, FederatedAggregator, GradientMessage};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       LEAF Federated Learning Benchmark                    ║");
    println!("║       Federated CfC with Trust-Weighted Aggregation        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let n_clients = 10;
    let n_rounds = 20;
    let samples_per_client = 100;
    let input_dim = 8;
    let hidden_dim = 32;
    let output_dim = 3;

    // ═══════════════════════════════════════════════════════════════
    // Test 1: Basic FedAvg Convergence
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "Test 1: FedAvg Convergence ({} clients, {} rounds)",
        n_clients, n_rounds
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let t = Instant::now();

    // Create global model
    let config = CfCNetworkConfig {
        input_dim,
        hidden_dim,
        num_layers: 1,
        output_dim,
        cell_config: CfCConfig {
            input_dim,
            hidden_dim,
            use_backbone: false,
            backbone_layers: 0,
            backbone_dim: 0,
            activation: ActivationType::Tanh,
            tau_range: (0.1, 5.0),
            dropout: 0.0,
            online_learning: None,
            gradient_clip: 1.0,
        },
        residual: false,
        bidirectional: false,
        enable_online_learning: false,
        online_learning_config: Default::default(),
    };

    let global_model = CfCNetwork::new(config.clone());
    let global_weights = global_model.get_weights();
    let weight_dim = global_weights.len();
    println!("  Model weight dimension: {}", weight_dim);

    // Create aggregator
    let mut aggregator = FederatedAggregator::new(global_weights.clone());

    // Generate non-IID client data (each client has different class distribution)
    let client_data = generate_non_iid_data(n_clients, samples_per_client, input_dim, output_dim);

    let mut round_losses: Vec<f64> = Vec::new();

    for round in 0..n_rounds {
        // Each client trains locally and submits gradients
        for (client_id, client_datum) in client_data.iter().enumerate().take(n_clients) {
            let mut local_model = CfCNetwork::new(config.clone());
            local_model.set_weights(aggregator.local_weights());

            // Train locally
            let (inputs, targets) = client_datum;
            let dt = 0.1;
            let lr = 0.01;
            for (inp, tgt) in inputs.iter().zip(targets.iter()) {
                let _ = local_model.train_step(inp, tgt, dt, lr);
            }

            // Compute gradient (weight difference)
            let local_weights = local_model.get_weights();
            let gradient: Vec<f32> = local_weights
                .iter()
                .zip(aggregator.local_weights().iter())
                .map(|(l, g)| l - g)
                .collect();

            // Trust score based on client reliability (simulated)
            let trust = 0.5 + 0.05 * client_id as f32;

            let mut source_id = [0u8; 32];
            source_id[0] = client_id as u8;

            let msg = GradientMessage::new(source_id, gradient, trust.min(1.0))
                .with_sample_count(samples_per_client as u64);

            aggregator.receive_gradient(msg);
        }

        // Aggregate and update global model
        if aggregator.aggregate_and_apply() {
            // Evaluate global model
            let mut eval_model = CfCNetwork::new(config.clone());
            eval_model.set_weights(aggregator.local_weights());

            let mut total_loss = 0.0f64;
            let mut count = 0;
            for (inputs, targets) in client_data.iter().take(n_clients) {
                for (inp, tgt) in inputs.iter().zip(targets.iter()) {
                    eval_model.reset();
                    let out = eval_model.forward(inp, 0.1);
                    let loss: f32 = out
                        .iter()
                        .zip(tgt.iter())
                        .map(|(o, t)| (o - t).powi(2))
                        .sum::<f32>()
                        / output_dim as f32;
                    total_loss += loss as f64;
                    count += 1;
                }
            }

            let avg_loss = total_loss / count as f64;
            round_losses.push(avg_loss);

            if round % 5 == 0 || round == n_rounds - 1 {
                println!("  Round {:2}: avg loss = {:.4}", round, avg_loss);
            }
        }
    }

    let elapsed = t.elapsed().as_secs_f64();
    let converged =
        round_losses.len() >= 2 && round_losses.last().unwrap() < round_losses.first().unwrap();
    println!("  FedAvg converged: {} ({:.1}s)", converged, elapsed);

    // ═══════════════════════════════════════════════════════════════
    // Test 2: Differential Privacy Impact
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: Differential Privacy Impact");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let dp_configs = [
        ("No DP", None),
        (
            "Low privacy (ε~10)",
            Some(DifferentialPrivacyConfig::low_privacy()),
        ),
        (
            "Moderate (ε~1)",
            Some(DifferentialPrivacyConfig::moderate_privacy()),
        ),
        (
            "High privacy (ε~0.1)",
            Some(DifferentialPrivacyConfig::high_privacy()),
        ),
    ];

    let mut dp_losses = Vec::new();

    for (name, dp_config) in &dp_configs {
        let mut agg = FederatedAggregator::new(global_weights.clone());
        if let Some(dp) = dp_config {
            agg = agg.with_differential_privacy(*dp);
        }

        // Run 10 rounds
        for _ in 0..10 {
            for (client_id, client_datum) in client_data.iter().enumerate().take(n_clients) {
                let mut local_model = CfCNetwork::new(config.clone());
                local_model.set_weights(agg.local_weights());

                let (inputs, targets) = client_datum;
                let dt = 0.1;
                for (inp, tgt) in inputs.iter().zip(targets.iter()) {
                    let _ = local_model.train_step(inp, tgt, dt, 0.01);
                }

                let local_weights = local_model.get_weights();
                let gradient: Vec<f32> = local_weights
                    .iter()
                    .zip(agg.local_weights().iter())
                    .map(|(l, g)| l - g)
                    .collect();

                let mut source_id = [0u8; 32];
                source_id[0] = client_id as u8;
                let msg = GradientMessage::new(source_id, gradient, 0.8)
                    .with_sample_count(samples_per_client as u64);
                agg.receive_gradient(msg);
            }
            agg.aggregate_and_apply();
        }

        // Evaluate
        let mut eval_model = CfCNetwork::new(config.clone());
        eval_model.set_weights(agg.local_weights());
        let mut total_loss = 0.0f64;
        let mut count = 0;
        for (inputs, targets) in client_data.iter().take(n_clients) {
            for (inp, tgt) in inputs.iter().zip(targets.iter()) {
                eval_model.reset();
                let out = eval_model.forward(inp, 0.1);
                let loss: f32 = out
                    .iter()
                    .zip(tgt.iter())
                    .map(|(o, t)| (o - t).powi(2))
                    .sum::<f32>()
                    / output_dim as f32;
                total_loss += loss as f64;
                count += 1;
            }
        }
        let avg_loss = total_loss / count as f64;
        dp_losses.push(avg_loss);
        println!("  {:<22} loss = {:.4}", name, avg_loss);
    }

    let dp_utility_preserved = dp_losses[1] < dp_losses[0] * 2.0; // Low DP should not double loss

    // ═══════════════════════════════════════════════════════════════
    // Test 3: Byzantine Fault Tolerance
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: Byzantine Fault Tolerance");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Run with 20% Byzantine clients sending garbage gradients
    let n_byzantine = 2; // 2 out of 10 clients

    let mut agg_no_bft = FederatedAggregator::new(global_weights.clone());
    let mut agg_with_bft =
        FederatedAggregator::new(global_weights.clone()).with_byzantine_tolerance(0.2);

    for _ in 0..10 {
        for (client_id, client_datum) in client_data.iter().enumerate().take(n_clients) {
            let gradient: Vec<f32> = if client_id < n_byzantine {
                // Byzantine: random large gradients
                let mut rng_seed = (client_id as u64 + 1) * 999;
                (0..weight_dim)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((rng_seed >> 33) as f32 / (1u64 << 31) as f32) * 10.0 - 5.0
                    })
                    .collect()
            } else {
                // Honest: small realistic gradients
                let mut local_model = CfCNetwork::new(config.clone());
                local_model.set_weights(agg_no_bft.local_weights());
                let (inputs, targets) = client_datum;
                for (inp, tgt) in inputs.iter().zip(targets.iter()) {
                    let _ = local_model.train_step(inp, tgt, 0.1, 0.01);
                }
                local_model
                    .get_weights()
                    .iter()
                    .zip(agg_no_bft.local_weights().iter())
                    .map(|(l, g)| l - g)
                    .collect()
            };

            let mut source_id = [0u8; 32];
            source_id[0] = client_id as u8;
            let msg1 = GradientMessage::new(source_id, gradient.clone(), 0.8);
            let msg2 = GradientMessage::new(source_id, gradient, 0.8);
            agg_no_bft.receive_gradient(msg1);
            agg_with_bft.receive_gradient(msg2);
        }
        agg_no_bft.aggregate_and_apply();
        agg_with_bft.aggregate_and_apply();
    }

    // Evaluate both
    let loss_no_bft = evaluate_model(
        &config,
        agg_no_bft.local_weights(),
        &client_data,
        output_dim,
    );
    let loss_with_bft = evaluate_model(
        &config,
        agg_with_bft.local_weights(),
        &client_data,
        output_dim,
    );

    println!("  Without BFT: loss = {:.4}", loss_no_bft);
    println!("  With BFT:    loss = {:.4}", loss_with_bft);

    let bft_helps = loss_with_bft < loss_no_bft;
    println!("  BFT improves robustness: {}", bft_helps);

    // ═══════════════════════════════════════════════════════════════
    // Test 4: Trust-Weighted Aggregation
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 4: Trust-Weighted vs Equal-Weight Aggregation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Mix of high-quality and low-quality clients
    let mut agg_trusted = FederatedAggregator::new(global_weights.clone());
    let mut agg_equal = FederatedAggregator::new(global_weights.clone());

    for _ in 0..10 {
        for (client_id, client_datum) in client_data.iter().enumerate().take(n_clients) {
            let mut local_model = CfCNetwork::new(config.clone());
            local_model.set_weights(agg_trusted.local_weights());

            let (inputs, targets) = client_datum;
            // Some clients train with more noise
            let lr = if client_id < 3 { 0.001 } else { 0.01 };
            for (inp, tgt) in inputs.iter().zip(targets.iter()) {
                let _ = local_model.train_step(inp, tgt, 0.1, lr);
            }

            let gradient: Vec<f32> = local_model
                .get_weights()
                .iter()
                .zip(agg_trusted.local_weights().iter())
                .map(|(l, g)| l - g)
                .collect();

            let mut source_id = [0u8; 32];
            source_id[0] = client_id as u8;

            // Trust-weighted: good clients get higher trust
            let trust = if client_id < 3 { 0.3 } else { 0.9 };
            let msg1 = GradientMessage::new(source_id, gradient.clone(), trust);
            // Equal weight
            let msg2 = GradientMessage::new(source_id, gradient, 0.5);

            agg_trusted.receive_gradient(msg1);
            agg_equal.receive_gradient(msg2);
        }
        agg_trusted.aggregate_and_apply();
        agg_equal.aggregate_and_apply();
    }

    let loss_trusted = evaluate_model(
        &config,
        agg_trusted.local_weights(),
        &client_data,
        output_dim,
    );
    let loss_equal = evaluate_model(&config, agg_equal.local_weights(), &client_data, output_dim);

    println!("  Trust-weighted: loss = {:.4}", loss_trusted);
    println!("  Equal-weight:   loss = {:.4}", loss_equal);

    let trust_helps = loss_trusted <= loss_equal * 1.1; // Trust should be at least comparable

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                 VALIDATION SUMMARY                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    let checks = vec![
        ("FedAvg converges over rounds", converged),
        ("DP preserves utility (low ε)", dp_utility_preserved),
        ("BFT improves Byzantine robustness", bft_helps),
        ("Trust-weighting is effective", trust_helps),
        (
            "Aggregation completes all rounds",
            round_losses.len() == n_rounds,
        ),
    ];

    let mut passed = 0;
    for (name, pass) in &checks {
        println!("║  {} {:50}   ║", if *pass { "PASS" } else { "FAIL" }, name);
        if *pass {
            passed += 1;
        }
    }
    println!("╟──────────────────────────────────────────────────────────────╢");
    println!(
        "║  Result: {}/{} tests passed                                 ║",
        passed,
        checks.len()
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Save
    let result_json = serde_json::json!({
        "benchmark": "LEAF Federated Learning (Synthetic Non-IID)",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "n_clients": n_clients,
        "n_rounds": n_rounds,
        "fedavg_converged": converged,
        "dp_utility_preserved": dp_utility_preserved,
        "bft_effective": bft_helps,
        "trust_weighting_effective": trust_helps,
        "tests_passed": passed,
        "tests_total": checks.len(),
    });

    std::fs::create_dir_all("data/benchmarks/federated").ok();
    if let Ok(f) = std::fs::File::create("data/benchmarks/federated/results.json") {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("Results saved to data/benchmarks/federated/results.json");
    }
}

#[allow(clippy::type_complexity)]
fn evaluate_model(
    config: &CfCNetworkConfig,
    weights: &[f32],
    client_data: &[(Vec<Array1<f32>>, Vec<Array1<f32>>)],
    output_dim: usize,
) -> f64 {
    let mut eval_model = CfCNetwork::new(config.clone());
    eval_model.set_weights(weights);
    let mut total_loss = 0.0f64;
    let mut count = 0;
    for (inputs, targets) in client_data {
        for (inp, tgt) in inputs.iter().zip(targets.iter()) {
            eval_model.reset();
            let out = eval_model.forward(inp, 0.1);
            let loss: f32 = out
                .iter()
                .zip(tgt.iter())
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f32>()
                / output_dim as f32;
            total_loss += loss as f64;
            count += 1;
        }
    }
    total_loss / count as f64
}

/// Generate non-IID client data where each client has biased class distributions
#[allow(clippy::type_complexity)]
fn generate_non_iid_data(
    n_clients: usize,
    samples_per_client: usize,
    input_dim: usize,
    output_dim: usize,
) -> Vec<(Vec<Array1<f32>>, Vec<Array1<f32>>)> {
    let mut client_data = Vec::new();

    for client_id in 0..n_clients {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        let mut rng = (client_id as u64 + 1) * 12345;
        let mut rand = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f32 / (1u64 << 31) as f32
        };

        // Each client is biased toward a particular class
        let primary_class = client_id % output_dim;

        for _ in 0..samples_per_client {
            // 70% chance of primary class, 30% other
            let class = if rand() < 0.7 {
                primary_class
            } else {
                (rand() * output_dim as f32) as usize % output_dim
            };

            // Generate input features correlated with class
            let input: Vec<f32> = (0..input_dim)
                .map(|d| {
                    let class_signal = ((class as f32 + d as f32) * 0.3).sin() * 0.5;
                    class_signal + (rand() - 0.5) * 0.3
                })
                .collect();

            // One-hot-ish target (soft labels)
            let target: Vec<f32> = (0..output_dim)
                .map(|c| if c == class { 0.8 } else { 0.1 })
                .collect();

            inputs.push(Array1::from_vec(input));
            targets.push(Array1::from_vec(target));
        }

        client_data.push((inputs, targets));
    }

    client_data
}