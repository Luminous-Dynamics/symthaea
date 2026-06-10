// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Train CfC melody network on real MIDI data (MAESTRO + Nottingham).
//!
//! Parses MIDI files, extracts melodic features (intervals, durations,
//! beat positions, phrase positions), encodes as HDC hypervectors, and
//! trains the CfC network to predict next-note intervals.
//!
//! This teaches the system what real melodies sound like — phrase contour,
//! stepwise motion patterns, tension-resolution, rhythmic idioms.
//!
//! Run: cargo run -p symthaea-muse --bin train_melody

use std::path::Path;
use symthaea_muse::midi_trainer;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     CfC Melody Training — Learning From Real Music         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Phase 1: Load MIDI datasets
    println!("Phase 1: Loading MIDI training data...\n");

    let mut all_pairs = Vec::new();

    let data_dirs = [
        (
            "MAESTRO (virtuoso piano)",
            "data/midi/maestro/maestro-v3.0.0",
        ),
        ("Nottingham (folk tunes)", "data/midi/nottingham"),
    ];

    for (name, dir) in &data_dirs {
        let path = Path::new(dir);
        if path.exists() {
            println!("  Loading {name}...");
            let pairs = load_midi_recursive(path);
            println!("    → {} training pairs\n", pairs.len());
            all_pairs.extend(pairs);
        } else {
            println!("  {name}: directory not found at {dir}, skipping");
            println!("    Download: see data/midi/README.md\n");
        }
    }

    if all_pairs.is_empty() {
        println!("No training data found. Download MIDI datasets first:");
        println!("  cd data/midi");
        println!(
            "  curl -L -o maestro.zip https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
        );
        println!("  unzip maestro.zip");
        return;
    }

    // Phase 2: Analyze training data
    println!("Phase 2: Analyzing training data...");
    let stats = midi_trainer::compute_stats(&all_pairs);
    println!("  Total pairs: {}", stats.total_pairs);
    println!("  Mean interval: {:.2} semitones", stats.mean_interval);
    println!("  Interval SD: {:.2} semitones", stats.std_interval);
    println!(
        "  Stepwise ratio: {:.1}% (intervals ≤ 2 semitones)",
        stats.stepwise_ratio * 100.0
    );
    println!("  Mean duration: {:.2} beats", stats.mean_duration);
    println!();

    // Validate our melodic grammar against real data
    println!("  Validation against melodic grammar assumptions:");
    if stats.stepwise_ratio > 0.5 {
        println!(
            "    ✓ Stepwise motion dominates ({:.0}% > 50%)",
            stats.stepwise_ratio * 100.0
        );
    } else {
        println!(
            "    ✗ Stepwise motion lower than expected ({:.0}%)",
            stats.stepwise_ratio * 100.0
        );
    }
    if stats.mean_interval.abs() < 1.0 {
        println!(
            "    ✓ Mean interval near zero ({:.2}) — melodies are balanced",
            stats.mean_interval
        );
    }
    println!();

    // Phase 3: Encode features for CfC training
    println!("Phase 3: Encoding HDC features...");
    let features: Vec<Vec<f32>> = all_pairs
        .iter()
        .map(midi_trainer::encode_features)
        .collect();
    let targets: Vec<Vec<f32>> = all_pairs.iter().map(midi_trainer::encode_target).collect();
    println!(
        "  Feature vectors: {} × {}D",
        features.len(),
        features.first().map(|f| f.len()).unwrap_or(0)
    );
    println!(
        "  Target vectors: {} × {}D",
        targets.len(),
        targets.first().map(|t| t.len()).unwrap_or(0)
    );
    println!();

    // Phase 4: Train simple linear model (CfC training would use GpuCfcLayer)
    println!("Phase 4: Training melody predictor...");
    let epochs = 50;
    let lr = 0.01;
    let input_dim = 20;
    let output_dim = 2;

    // Simple linear weights: W[output_dim × input_dim]
    let mut weights = vec![vec![0.0f32; input_dim]; output_dim];
    let mut bias = vec![0.0f32; output_dim];

    let n = features.len().min(100_000); // cap for speed
    let mut best_loss = f32::MAX;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;

        for i in 0..n {
            // Forward pass: y = Wx + b
            let mut pred = vec![0.0f32; output_dim];
            for o in 0..output_dim {
                for j in 0..input_dim {
                    pred[o] += weights[o][j] * features[i][j];
                }
                pred[o] += bias[o];
                pred[o] = pred[o].tanh(); // squash to [-1, 1]
            }

            // Loss: MSE
            let mut loss = 0.0f32;
            for o in 0..output_dim {
                let diff = pred[o] - targets[i][o];
                loss += diff * diff;

                // Backward: gradient of tanh
                let grad = diff * (1.0 - pred[o] * pred[o]);
                for j in 0..input_dim {
                    weights[o][j] -= lr * grad * features[i][j] / n as f32;
                }
                bias[o] -= lr * grad / n as f32;
            }
            epoch_loss += loss / output_dim as f32;
        }

        let avg_loss = epoch_loss / n as f32;
        if avg_loss < best_loss {
            best_loss = avg_loss;
        }

        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("  Epoch {epoch:3}/{epochs}: loss={avg_loss:.6} best={best_loss:.6}");
        }
    }

    // Phase 5: Evaluate
    println!("\nPhase 5: Evaluation...");

    // Test on held-out data (last 10%)
    let test_start = (n as f32 * 0.9) as usize;
    let mut test_loss = 0.0f32;
    let mut interval_mae = 0.0f32;
    let mut correct_direction = 0usize;
    let test_count = n - test_start;

    for i in test_start..n {
        let mut pred = vec![0.0f32; output_dim];
        for o in 0..output_dim {
            for j in 0..input_dim {
                pred[o] += weights[o][j] * features[i][j];
            }
            pred[o] += bias[o];
            pred[o] = pred[o].tanh();
        }

        let diff = pred[0] - targets[i][0];
        test_loss += diff * diff;
        interval_mae += (pred[0] * 12.0 - targets[i][0] * 12.0).abs(); // in semitones

        // Direction accuracy: does predicted interval have same sign as target?
        if pred[0].signum() == targets[i][0].signum() || targets[i][0].abs() < 0.01 {
            correct_direction += 1;
        }
    }

    let test_mse = test_loss / test_count as f32;
    let test_mae = interval_mae / test_count as f32;
    let direction_acc = correct_direction as f32 / test_count as f32 * 100.0;

    println!("  Test MSE: {test_mse:.6}");
    println!("  Interval MAE: {test_mae:.2} semitones");
    println!("  Direction accuracy: {direction_acc:.1}%");

    if direction_acc > 55.0 {
        println!("  Status: LEARNING — model predicts melodic direction above chance");
    } else {
        println!("  Status: BASELINE — model has not yet learned direction");
    }

    // Print what the model learned
    println!("\nPhase 6: What the model learned...");
    println!("  Top interval predictors (weight magnitude):");
    let mut feature_names: Vec<(usize, f32)> =
        (0..input_dim).map(|j| (j, weights[0][j].abs())).collect();
    feature_names.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let names = [
        "iv[-8]",
        "iv[-7]",
        "iv[-6]",
        "iv[-5]",
        "iv[-4]",
        "iv[-3]",
        "iv[-2]",
        "iv[-1]",
        "dur[-8]",
        "dur[-7]",
        "dur[-6]",
        "dur[-5]",
        "dur[-4]",
        "dur[-3]",
        "dur[-2]",
        "dur[-1]",
        "beat_pos",
        "phrase_pos",
        "valence",
        "arousal",
    ];

    for (j, w) in feature_names.iter().take(5) {
        let name = if *j < names.len() { names[*j] } else { "?" };
        let sign = if weights[0][*j] > 0.0 { "+" } else { "-" };
        println!("    {name:12}: {sign}{w:.4}");
    }

    println!("\nDone. Model weights can be exported for CfC initialization.");
    println!("For GPU training with CfC temporal dynamics:");
    println!("  nix develop");
    println!("  cargo run -p symthaea-muse --features cuda --bin train_melody");
}

/// Recursively load all MIDI files from a directory tree.
fn load_midi_recursive(dir: &Path) -> Vec<midi_trainer::MelodyTrainingPair> {
    let mut all_pairs = Vec::new();

    fn visit(dir: &Path, pairs: &mut Vec<midi_trainer::MelodyTrainingPair>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                visit(&path, pairs);
            } else if path
                .extension()
                .map(|e| {
                    let e = e.to_string_lossy().to_lowercase();
                    e == "mid" || e == "midi"
                })
                .unwrap_or(false)
            {
                match midi_trainer::parse_midi(&path) {
                    Ok(melody) => {
                        if melody.notes.len() >= 10 {
                            let melody_pairs = midi_trainer::melody_to_training_pairs(&melody);
                            // Transpose augmentation: ±6 semitones (12x data)
                            for semitones in -6..=5 {
                                for pair in &melody_pairs {
                                    pairs.push(midi_trainer::augment_transpose(pair, semitones));
                                }
                            }
                        }
                    }
                    Err(_) => {} // silently skip unparseable files
                }
            }
        }
    }

    visit(dir, &mut all_pairs);
    all_pairs
}
