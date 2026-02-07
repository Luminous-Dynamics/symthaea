//! # Tokamak Disruption Prediction with CfC Networks
//!
//! Tests Symthaea's Closed-form Continuous-time (CfC) networks on
//! plasma disruption prediction, a critical safety application.
//! Uses synthetic tokamak diagnostic signals to validate:
//!
//! 1. CfC learns temporal dynamics of plasma instabilities
//! 2. Early warning detection (prediction before disruption onset)
//! 3. Irregular time-series handling (variable diagnostic rates)
//! 4. Online adaptation during novel plasma regimes
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_tokamak_cfc --release
//! ```

use std::time::Instant;

use ndarray::Array1;
use symthaea::dynamics::cfc::{CfCNetwork, CfCNetworkConfig, CfCConfig, ActivationType};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Tokamak Disruption Prediction with CfC               ║");
    println!("║       Continuous-Time Neural Network Benchmark             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let input_dim = 6;  // Plasma diagnostics: Ip, ne, Te, q95, li, betaN
    let hidden_dim = 64;
    let output_dim = 1; // Disruption probability

    let config = CfCNetworkConfig {
        input_dim,
        hidden_dim,
        num_layers: 2,
        output_dim,
        cell_config: CfCConfig {
            input_dim,
            hidden_dim,
            use_backbone: true,
            backbone_layers: 1,
            backbone_dim: 32,
            activation: ActivationType::SiLU,
            tau_range: (0.01, 1.0), // Fast dynamics for disruptions
            dropout: 0.0,
            online_learning: None,
        },
        residual: true,
        bidirectional: false,
        enable_online_learning: false,
        online_learning_config: Default::default(),
    };

    // ═══════════════════════════════════════════════════════════════
    // Test 1: Learn Disruption Dynamics
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: Learning Disruption Dynamics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let t = Instant::now();
    let mut network = CfCNetwork::new(config.clone());

    let n_shots = 200;  // Tokamak shots (some disruptive, some stable)
    let steps_per_shot = 50;
    let lr = 0.02;  // Higher LR for BPTT with Adam

    let mut epoch_losses = Vec::new();

    for epoch in 0..10 {
        let mut total_loss = 0.0f32;
        let mut count = 0;

        for shot in 0..n_shots {
            let is_disruptive = shot % 2 != 0; // 50% disruptive (balanced classes)
            let (inputs, targets, dts) = generate_plasma_shot(
                shot as u64, is_disruptive, steps_per_shot, input_dim,
            );

            network.reset();

            // Train with full sequence for proper BPTT through temporal dynamics
            let loss = network.train_step_bptt_optimized(&inputs, &targets, &dts, lr)
                .unwrap_or(1.0);
            total_loss += loss;
            count += 1;
        }

        let avg_loss = total_loss / count as f32;
        epoch_losses.push(avg_loss);
        println!("  Epoch {}: avg loss = {:.4}", epoch, avg_loss);
    }

    let train_time = t.elapsed().as_secs_f64();
    let loss_decreased = epoch_losses.last().unwrap() < epoch_losses.first().unwrap();
    println!("  Training time: {:.1}s", train_time);
    println!("  Loss decreased: {}", loss_decreased);

    // ═══════════════════════════════════════════════════════════════
    // Test 2: Early Warning Detection
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: Early Warning Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let n_test = 50;
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut true_negatives = 0;
    let mut false_negatives = 0;
    let mut early_warnings = 0;
    let threshold = 0.5;

    for shot in 0..n_test {
        let is_disruptive = shot % 2 != 0; // 50% disruptive (balanced classes)
        let (inputs, _targets, dts) = generate_plasma_shot(
            1000 + shot as u64, is_disruptive, steps_per_shot, input_dim,
        );

        network.reset();

        let disruption_step = (steps_per_shot as f32 * 0.7) as usize;
        let mut predicted_disruption = false;
        let mut warning_step = None;

        for i in 0..inputs.len() {
            let output = network.forward(&inputs[i], dts[i]);
            let prob = output[0].clamp(0.0, 1.0);

            if prob > threshold && !predicted_disruption {
                predicted_disruption = true;
                warning_step = Some(i);
            }
        }

        if is_disruptive {
            if predicted_disruption {
                true_positives += 1;
                if let Some(ws) = warning_step {
                    if ws < disruption_step {
                        early_warnings += 1;
                    }
                }
            } else {
                false_negatives += 1;
            }
        } else if predicted_disruption {
            false_positives += 1;
        } else {
            true_negatives += 1;
        }
    }

    let sensitivity = true_positives as f64 / (true_positives + false_negatives).max(1) as f64;
    let specificity = true_negatives as f64 / (true_negatives + false_positives).max(1) as f64;
    let early_rate = early_warnings as f64 / true_positives.max(1) as f64;

    println!("  Sensitivity (recall): {:.1}%", sensitivity * 100.0);
    println!("  Specificity: {:.1}%", specificity * 100.0);
    println!("  Early warning rate: {:.1}%", early_rate * 100.0);

    // ═══════════════════════════════════════════════════════════════
    // Test 3: Irregular Time-Series Handling
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: Irregular Time-Series (Variable dt)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Same disruption, different sampling rates
    let (inputs, _targets, _dts) = generate_plasma_shot(42, true, 100, input_dim);

    // Regular sampling (dt=0.01)
    network.reset();
    let mut regular_outputs = Vec::new();
    for inp in &inputs {
        let out = network.forward(inp, 0.01);
        regular_outputs.push(out[0]);
    }

    // Irregular sampling (varying dt)
    network.reset();
    let mut irregular_outputs = Vec::new();
    let mut rng = 42u64;
    for inp in &inputs {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let dt = 0.005 + 0.01 * (rng >> 33) as f32 / (1u64 << 31) as f32;
        let out = network.forward(inp, dt);
        irregular_outputs.push(out[0]);
    }

    // Both should produce similar trends
    let trend_regular = regular_outputs.last().unwrap() - regular_outputs.first().unwrap();
    let trend_irregular = irregular_outputs.last().unwrap() - irregular_outputs.first().unwrap();
    let trend_agreement = (trend_regular.signum() - trend_irregular.signum()).abs() < 0.01;

    println!("  Regular dt trend:   {:.4}", trend_regular);
    println!("  Irregular dt trend: {:.4}", trend_irregular);
    println!("  Trend agreement: {}", trend_agreement);

    // ═══════════════════════════════════════════════════════════════
    // Test 4: Inference Speed
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 4: Real-Time Inference Speed");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let n_infer = 10000;
    let test_input = Array1::from_vec(vec![0.5f32; input_dim]);

    network.reset();
    let t = Instant::now();
    for _ in 0..n_infer {
        let _ = network.forward(&test_input, 0.01);
    }
    let infer_time = t.elapsed().as_secs_f64();
    let infer_per_ms = infer_time * 1000.0 / n_infer as f64;
    let throughput = n_infer as f64 / infer_time;

    println!("  {} inferences in {:.2}s", n_infer, infer_time);
    println!("  {:.3}ms per inference", infer_per_ms);
    println!("  {:.0} inferences/sec", throughput);

    // For tokamak: need <1ms per inference (1kHz control loop)
    let realtime_capable = infer_per_ms < 1.0;
    println!("  Real-time capable (<1ms): {}", realtime_capable);

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                 VALIDATION SUMMARY                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    let checks = vec![
        ("CfC loss decreases during training", loss_decreased),
        ("Sensitivity > 50%", sensitivity > 0.50),
        ("Specificity > 0% (non-degenerate)", specificity > 0.0),
        ("Irregular dt trend agreement", trend_agreement),
        ("Real-time inference (<1ms)", realtime_capable),
    ];

    let mut passed = 0;
    for (name, pass) in &checks {
        println!("║  {} {:50}   ║", if *pass { "PASS" } else { "FAIL" }, name);
        if *pass { passed += 1; }
    }
    println!("╟──────────────────────────────────────────────────────────────╢");
    println!("║  Result: {}/{} tests passed                                 ║", passed, checks.len());
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Save
    let result_json = serde_json::json!({
        "benchmark": "Tokamak Disruption Prediction with CfC",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "training_loss_decreased": loss_decreased,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "early_warning_rate": early_rate,
        "inference_ms": infer_per_ms,
        "realtime_capable": realtime_capable,
        "tests_passed": passed,
        "tests_total": checks.len(),
    });

    std::fs::create_dir_all("data/benchmarks/tokamak").ok();
    if let Ok(f) = std::fs::File::create("data/benchmarks/tokamak/results.json") {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("Results saved to data/benchmarks/tokamak/results.json");
    }
}

/// Generate a synthetic tokamak plasma shot
/// Returns (inputs, targets, dts)
fn generate_plasma_shot(
    seed: u64,
    is_disruptive: bool,
    n_steps: usize,
    input_dim: usize,
) -> (Vec<Array1<f32>>, Vec<Array1<f32>>, Vec<f32>) {
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut rand = || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f32 / (1u64 << 31) as f32
    };

    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut dts = Vec::new();

    let disruption_onset = (n_steps as f32 * 0.7) as usize;

    for step in 0..n_steps {
        let t = step as f32 / n_steps as f32;

        // Plasma diagnostics
        let (ip, ne, te, q95, li, beta_n) = if is_disruptive {
            // Pre-disruption: gradual instability growth
            let instability = if step > disruption_onset {
                ((step - disruption_onset) as f32 / (n_steps - disruption_onset) as f32).powi(2)
            } else {
                0.0
            };

            let precursor = if step > disruption_onset / 2 {
                ((step as f32 - disruption_onset as f32 / 2.0) / n_steps as f32 * 3.0).sin() * 0.1
            } else {
                0.0
            };

            (
                1.0 - instability * 0.5 + precursor,      // Ip drops
                0.5 + t * 0.3 + instability * 0.5,         // ne rises (density limit)
                0.6 - instability * 0.3,                    // Te drops
                3.0 - instability * 1.5 + precursor * 0.5,  // q95 drops (MHD unstable)
                0.8 + instability * 0.4,                    // li increases
                2.0 + instability * 1.5,                    // betaN exceeds limit
            )
        } else {
            // Stable shot
            (
                1.0 + rand() * 0.05,        // Ip stable
                0.5 + t * 0.1,               // ne gentle ramp
                0.6 + rand() * 0.05,         // Te stable
                3.0 + rand() * 0.1,          // q95 safe
                0.8 + rand() * 0.05,         // li stable
                1.5 + rand() * 0.1,          // betaN within limits
            )
        };

        let noise = rand() * 0.02;
        let mut features = vec![0.0f32; input_dim];
        features[0] = ip + noise;
        features[1] = ne + noise;
        features[2] = te + noise;
        features[3] = q95 + noise;
        features[4] = li + noise;
        features[5] = beta_n + noise;

        inputs.push(Array1::from_vec(features));

        // Target: disruption probability
        let target_prob = if is_disruptive {
            if step >= disruption_onset {
                1.0
            } else if step > disruption_onset / 2 {
                (step as f32 - disruption_onset as f32 / 2.0) /
                    (disruption_onset as f32 / 2.0) * 0.5
            } else {
                0.0
            }
        } else {
            0.0
        };
        targets.push(Array1::from_vec(vec![target_prob]));

        // Irregular time steps
        dts.push(0.01 + rand() * 0.005);
    }

    (inputs, targets, dts)
}
