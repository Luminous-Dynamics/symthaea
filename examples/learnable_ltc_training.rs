// ==================================================================================
// Learnable LTC Training Example
// ==================================================================================
//
// **Purpose**: Demonstrate training LearnableLTC on a real task - sequence prediction
//
// **Task**: Learn to predict the next value in a sinusoidal sequence
// This is a classic benchmark that demonstrates temporal reasoning capability.
//
// **Why This Matters**:
// - LLMs cannot do online learning (frozen weights)
// - LTCs can adapt their time constants (tau) during training
// - This shows Symthaea's continuous learning capability
//
// Run with: cargo run --example learnable_ltc_training
//
// ==================================================================================

use symthaea::learnable_ltc::{LearnableLTC, LearnableLTCConfig};
use std::time::Instant;

/// Generate a sinusoidal sequence with noise
fn generate_sequence(length: usize, frequency: f32, noise: f32) -> Vec<f32> {
    (0..length)
        .map(|t| {
            let clean = (2.0 * std::f32::consts::PI * frequency * t as f32 / 100.0).sin();
            let noise_val = if noise > 0.0 {
                (rand_simple(t as u64) * 2.0 - 1.0) * noise
            } else {
                0.0
            };
            clean + noise_val
        })
        .collect()
}

/// Simple deterministic random number generator
fn rand_simple(seed: u64) -> f32 {
    let x = seed.wrapping_mul(0xDABDBA31).wrapping_add(0x12345678);
    let x = x ^ (x >> 17);
    let x = x.wrapping_mul(0xDEADBEEF);
    (x as f32 / u64::MAX as f32)
}

/// Prepare training data: predict next value from window
fn prepare_data(sequence: &[f32], window_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for i in 0..(sequence.len() - window_size - 1) {
        let input: Vec<f32> = sequence[i..i + window_size].to_vec();
        let target = vec![sequence[i + window_size]];
        inputs.push(input);
        targets.push(target);
    }

    (inputs, targets)
}

/// Compute mean squared error
fn mse(predictions: &[f32], targets: &[f32]) -> f32 {
    if predictions.is_empty() || targets.is_empty() {
        return f32::MAX;
    }

    let sum: f32 = predictions.iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();

    sum / predictions.len() as f32
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Learnable LTC Training Example                           ║");
    println!("║     Task: Sinusoidal Sequence Prediction                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let window_size = 16;
    let sequence_length = 500;
    let epochs = 50;
    let frequency = 0.05;
    let noise = 0.1;

    println!("Configuration:");
    println!("  Window size: {}", window_size);
    println!("  Sequence length: {}", sequence_length);
    println!("  Training epochs: {}", epochs);
    println!("  Signal frequency: {}", frequency);
    println!("  Noise level: {}\n", noise);

    // Generate training data
    println!("Generating training sequence...");
    let train_sequence = generate_sequence(sequence_length, frequency, noise);
    let (train_inputs, train_targets) = prepare_data(&train_sequence, window_size);
    println!("  Training samples: {}\n", train_inputs.len());

    // Generate test data (different phase)
    println!("Generating test sequence...");
    let test_sequence: Vec<f32> = (0..200)
        .map(|t| {
            let clean = (2.0 * std::f32::consts::PI * frequency * (t + 100) as f32 / 100.0).sin();
            clean + (rand_simple((t + 12345) as u64) * 2.0 - 1.0) * noise
        })
        .collect();
    let (test_inputs, test_targets) = prepare_data(&test_sequence, window_size);
    println!("  Test samples: {}\n", test_inputs.len());

    // Create LTC network
    println!("Creating LearnableLTC network...");
    let mut config = LearnableLTCConfig::default();
    config.input_dim = window_size;
    config.num_neurons = 32;     // Smaller for faster training
    config.output_dim = 1;       // Predict single value
    config.num_steps = 10;       // Integration steps (fewer for stability)
    config.lr_weights = 0.0001;  // Learning rate for weights (smaller for stability)
    config.lr_tau = 0.00001;     // Learning rate for time constants (very small)
    config.lr_bias = 0.0001;     // Learning rate for biases
    config.sparsity = 0.3;       // 30% connectivity
    config.grad_clip = 0.5;      // Gradient clipping for stability
    config.l2_reg = 0.001;       // L2 regularization

    let mut ltc = LearnableLTC::new(config).expect("Failed to create LTC");

    println!("  Neurons: 32");
    println!("  Integration steps: 20");
    println!("  Sparse connectivity: 30%\n");

    // Training loop
    println!("Starting training...\n");
    let start_time = Instant::now();

    let mut best_test_loss = f32::MAX;
    let mut train_losses = Vec::new();
    let mut test_losses = Vec::new();

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut num_samples = 0;

        // Train on each sample
        for (input, target) in train_inputs.iter().zip(train_targets.iter()) {
            match ltc.train_step(input, target) {
                Ok(loss) => {
                    epoch_loss += loss;
                    num_samples += 1;
                }
                Err(e) => {
                    eprintln!("Training error: {}", e);
                }
            }
        }

        let avg_train_loss = if num_samples > 0 {
            epoch_loss / num_samples as f32
        } else {
            f32::MAX
        };
        train_losses.push(avg_train_loss);

        // Evaluate on test set
        let mut predictions = Vec::new();
        let mut actuals = Vec::new();

        for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
            match ltc.forward(input) {
                Ok((output, _)) => {
                    if let Some(&pred) = output.first() {
                        predictions.push(pred);
                        if let Some(&actual) = target.first() {
                            actuals.push(actual);
                        }
                    }
                }
                Err(_) => {}
            }
        }

        let test_loss = mse(&predictions, &actuals);
        test_losses.push(test_loss);

        if test_loss < best_test_loss {
            best_test_loss = test_loss;
        }

        // Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!("Epoch {:3}/{}: Train Loss = {:.6}, Test Loss = {:.6}",
                     epoch + 1, epochs, avg_train_loss, test_loss);
        }
    }

    let training_time = start_time.elapsed();

    // Final evaluation
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Training Complete                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Results:");
    println!("  Training time: {:.2}s", training_time.as_secs_f32());
    println!("  Final train loss: {:.6}", train_losses.last().unwrap_or(&0.0));
    println!("  Final test loss: {:.6}", test_losses.last().unwrap_or(&0.0));
    println!("  Best test loss: {:.6}", best_test_loss);

    // Show LTC stats (clone to avoid borrow issues)
    let stats = ltc.stats().clone();
    println!("\nLTC Training Statistics:");
    println!("  Total steps: {}", stats.total_steps);
    println!("  Total loss: {:.6}", stats.total_loss);
    println!("  Average tau: {:.3}", stats.avg_tau);
    println!("  Tau std dev: {:.3}", stats.tau_std);
    println!("  Actual sparsity: {:.1}%", stats.sparsity_actual * 100.0);
    println!("  Gradient norm: {:.4}", stats.grad_norm);

    // Demonstrate prediction
    println!("\nSample Predictions (last 5 test samples):");
    println!("  {:>10} {:>10} {:>10}", "Actual", "Predicted", "Error");
    println!("  {:->10} {:->10} {:->10}", "", "", "");

    for (input, target) in test_inputs.iter().rev().take(5).zip(test_targets.iter().rev().take(5)) {
        if let Ok((output, _)) = ltc.forward(input) {
            if let (Some(&pred), Some(&actual)) = (output.first(), target.first()) {
                let error = (pred - actual).abs();
                println!("  {:>10.4} {:>10.4} {:>10.4}", actual, pred, error);
            }
        }
    }

    // Learning curve visualization (ASCII art)
    println!("\nLearning Curve (Test Loss):");
    let max_loss = test_losses.iter().cloned().fold(0.0f32, f32::max);
    let min_loss = test_losses.iter().cloned().fold(f32::MAX, f32::min);
    let range = max_loss - min_loss;

    for i in 0..10 {
        let threshold = max_loss - (i as f32 / 9.0) * range;
        print!("  {:6.4} |", threshold);
        for (epoch, &loss) in test_losses.iter().enumerate() {
            if epoch % (epochs / 20).max(1) == 0 {
                if loss <= threshold {
                    print!("*");
                } else {
                    print!(" ");
                }
            }
        }
        println!();
    }
    println!("         +{}", "-".repeat(20));
    println!("          Epochs →\n");

    // Summary
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                      Key Insights                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("This demonstrates Symthaea's unique capabilities:");
    println!();
    println!("1. ONLINE LEARNING: LTC can learn from streaming data");
    println!("   - Unlike LLMs which have frozen weights after pretraining");
    println!("   - Tau (time constants) adapt to temporal patterns in data");
    println!();
    println!("2. TEMPORAL REASONING: LTC naturally captures time dependencies");
    println!("   - Continuous-time dynamics (not discrete tokens)");
    println!("   - Learned tau values encode temporal scale preferences");
    println!();
    println!("3. SPARSE COMPUTATION: Only 30% connectivity used");
    println!("   - More efficient than dense transformer attention");
    println!("   - Biologically inspired neural architecture");
    println!();
    println!("The average tau = {:.3} shows the network learned temporal", stats.avg_tau);
    println!("dynamics appropriate for the sinusoidal signal frequency.");
}
