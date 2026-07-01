// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Probe Training Tool
//!
//! Trains a linear projection (W matrix) from embedding space to HDC space.
//! This enables Neural Bridge to project LLM activations to consciousness topology.
//!
//! ## Algorithm
//!
//! 1. For each concept in training set:
//!    - Extract embedding via BGE-M3 (or other encoder)
//!    - Generate target HDC vector (deterministic from concept text)
//!
//! 2. Train W matrix via gradient descent:
//!    - Loss: 1 - cosine_similarity(W @ embedding, target_hdc)
//!    - Optimizer: SGD with momentum
//!
//! 3. Save W as .npy file
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example train_probe --features neural-bridge -- \
//!     --concepts data/consciousness_probe/training_concepts.json \
//!     --output models/neural_bridge/probe_weights_custom.npy \
//!     --epochs 1000 \
//!     --lr 0.01
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::bge_m3::{BGE_M3_DIM, BgeM3};

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::{HDC_DIMENSION, binary_hv::BinaryHV};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This tool requires the 'neural-bridge' feature.");
        println!("Run with: cargo run --example train_probe --features neural-bridge");
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_training()
}

#[cfg(feature = "neural-bridge")]
fn run_training() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   NEURAL BRIDGE PROBE TRAINER");
    println!("   Training: {} → {} projection", BGE_M3_DIM, HDC_DIMENSION);
    println!("================================================================\n");

    // Training configuration
    let config = TrainingConfig {
        learning_rate: 0.01,
        momentum: 0.9,
        epochs: 500,
        batch_size: 32,
        output_path: PathBuf::from("models/neural_bridge/probe_weights_trained.npy"),
        validation_split: 0.2,
    };

    // Training concepts (mix of phenomenal and functional for balanced representation)
    let training_concepts = get_training_concepts();
    println!("Training concepts: {}\n", training_concepts.len());

    // Load encoder
    println!("Loading BGE-M3 encoder...");
    let load_start = Instant::now();
    let encoder = BgeM3::load_from_hub("BAAI/bge-m3")?;
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    // Extract embeddings
    println!("Extracting embeddings...");
    let extract_start = Instant::now();
    let embeddings: Vec<Vec<f32>> = training_concepts
        .iter()
        .map(|concept| encoder.encode(concept))
        .collect::<Result<Vec<_>>>()?;
    println!(
        "  Extracted {} embeddings in {:.2}s\n",
        embeddings.len(),
        extract_start.elapsed().as_secs_f64()
    );

    // Generate target HDC vectors
    println!("Generating target HDC vectors...");
    let targets: Vec<Vec<f32>> = training_concepts
        .iter()
        .enumerate()
        .map(|(i, concept)| {
            let seed = hash_concept(concept) + i as u64;
            let hv = BinaryHV::random(seed);
            hv_to_f32(&hv)
        })
        .collect();
    println!("  Generated {} targets\n", targets.len());

    // Split into train/validation
    let split_idx = (embeddings.len() as f32 * (1.0 - config.validation_split)) as usize;
    let (train_embeddings, val_embeddings) = embeddings.split_at(split_idx);
    let (train_targets, val_targets) = targets.split_at(split_idx);

    println!("Dataset split:");
    println!("  Training: {}", train_embeddings.len());
    println!("  Validation: {}\n", val_embeddings.len());

    // Initialize weights with Xavier initialization
    println!("Initializing weights...");
    let mut weights = initialize_weights(BGE_M3_DIM, HDC_DIMENSION);
    let mut velocity = vec![0.0f32; BGE_M3_DIM * HDC_DIMENSION];
    println!(
        "  Weight matrix: {}x{} ({:.2}MB)\n",
        HDC_DIMENSION,
        BGE_M3_DIM,
        (weights.len() * 4) as f64 / 1_000_000.0
    );

    // Training loop
    println!("================================================================");
    println!("   TRAINING");
    println!("================================================================\n");

    let train_start = Instant::now();
    let mut best_val_loss = f32::MAX;

    for epoch in 0..config.epochs {
        // Training step
        let train_loss = train_epoch(
            &mut weights,
            &mut velocity,
            train_embeddings,
            train_targets,
            config.learning_rate,
            config.momentum,
        );

        // Validation step
        let val_loss = compute_loss(&weights, val_embeddings, val_targets);

        // Progress logging
        if epoch % 50 == 0 || epoch == config.epochs - 1 {
            let elapsed = train_start.elapsed().as_secs_f64();
            println!(
                "Epoch {:4}/{}: train_loss={:.6}, val_loss={:.6}, elapsed={:.1}s",
                epoch + 1,
                config.epochs,
                train_loss,
                val_loss,
                elapsed
            );
        }

        // Track best model
        if val_loss < best_val_loss {
            best_val_loss = val_loss;
        }
    }

    println!(
        "\nTraining completed in {:.1}s",
        train_start.elapsed().as_secs_f64()
    );
    println!("Best validation loss: {:.6}\n", best_val_loss);

    // Save weights
    println!("Saving weights to {}...", config.output_path.display());
    save_weights_npy(&weights, HDC_DIMENSION, BGE_M3_DIM, &config.output_path)?;
    println!("  Saved successfully!\n");

    // Final evaluation
    println!("================================================================");
    println!("   EVALUATION");
    println!("================================================================\n");

    let final_train_loss = compute_loss(&weights, train_embeddings, train_targets);
    let final_val_loss = compute_loss(&weights, val_embeddings, val_targets);

    println!("Final losses:");
    println!(
        "  Training: {:.6} (cosine similarity: {:.4})",
        final_train_loss,
        1.0 - final_train_loss
    );
    println!(
        "  Validation: {:.6} (cosine similarity: {:.4})",
        final_val_loss,
        1.0 - final_val_loss
    );

    // Test a few examples
    println!("\nSample projections:");
    for (i, concept) in training_concepts.iter().take(3).enumerate() {
        let projected = project(&weights, &embeddings[i], HDC_DIMENSION);
        let similarity = cosine_similarity(&projected, &targets[i]);
        println!("  \"{}\": sim={:.4}", truncate(concept, 40), similarity);
    }

    println!("\n================================================================");
    println!("   TRAINING COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
struct TrainingConfig {
    learning_rate: f32,
    momentum: f32,
    epochs: usize,
    batch_size: usize,
    output_path: PathBuf,
    validation_split: f32,
}

#[cfg(feature = "neural-bridge")]
fn get_training_concepts() -> Vec<&'static str> {
    vec![
        // Phenomenal concepts
        "The subjective experience of seeing red",
        "What it is like to feel pain",
        "The taste of sweetness on my tongue",
        "Self-awareness that I am thinking",
        "The unified field of conscious awareness",
        "The felt quality of intense joy",
        "The raw sensation of pressure on my skin",
        "Awareness of my own awareness",
        "The phenomenal present moment",
        "The mysterious gap between brain and mind",
        "The experience of deep blue in the sky",
        "The feeling of warmth spreading through my body",
        "Consciousness of my body as my own",
        "The sense of being the author of my thoughts",
        "The qualitative nature of mental states",
        // Functional concepts
        "Recursive function evaluation",
        "Memory allocation and deallocation",
        "Binary search tree traversal",
        "Matrix multiplication operations",
        "Database query optimization",
        "Graph traversal using depth-first search",
        "Statistical hypothesis testing",
        "Network packet routing algorithms",
        "Compiler lexical analysis",
        "Eigenvalue decomposition of matrices",
        "Hash table collision resolution",
        "Dynamic programming optimization",
        "Operating system process scheduling",
        "Distributed consensus algorithms",
        "Control system feedback loop tuning",
        // Additional mixed concepts
        "The concept of infinity in mathematics",
        "Understanding another person's emotions",
        "The feeling of time passing slowly",
        "Abstract reasoning about logical propositions",
        "The sensation of music creating emotion",
        "Pattern recognition in visual scenes",
        "Memory consolidation during sleep",
        "Decision making under uncertainty",
        "Language comprehension mechanisms",
        "Spatial navigation and mental maps",
    ]
}

#[cfg(feature = "neural-bridge")]
fn hash_concept(text: &str) -> u64 {
    text.bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
}

#[cfg(feature = "neural-bridge")]
fn hv_to_f32(hv: &BinaryHV) -> Vec<f32> {
    let bipolar = hv.to_bipolar();
    bipolar.iter().map(|&b| b as f32).collect()
}

#[cfg(feature = "neural-bridge")]
fn initialize_weights(input_dim: usize, output_dim: usize) -> Vec<f32> {
    // Xavier initialization
    let std = (2.0 / (input_dim + output_dim) as f64).sqrt() as f32;
    let mut weights = Vec::with_capacity(output_dim * input_dim);
    let mut seed: u64 = 42;

    for _ in 0..(output_dim * input_dim) {
        // Simple LCG for reproducibility
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (seed as f32) / (u64::MAX as f32);
        // Box-Muller for normal distribution
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = (seed as f32) / (u64::MAX as f32);
        let normal = (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos();
        weights.push(normal * std);
    }

    weights
}

#[cfg(feature = "neural-bridge")]
fn train_epoch(
    weights: &mut [f32],
    velocity: &mut [f32],
    embeddings: &[Vec<f32>],
    targets: &[Vec<f32>],
    lr: f32,
    momentum: f32,
) -> f32 {
    let output_dim = HDC_DIMENSION;
    let input_dim = BGE_M3_DIM;
    let mut total_loss = 0.0;

    for (embedding, target) in embeddings.iter().zip(targets.iter()) {
        // Forward pass: projected = W @ embedding
        let projected = project(weights, embedding, output_dim);

        // Compute loss and gradients
        let loss = 1.0 - cosine_similarity(&projected, target);
        total_loss += loss;

        // Compute gradient of cosine loss
        // d(1 - cos(a,b))/dW = -d(cos(a,b))/dW
        let proj_norm = norm(&projected);
        let tgt_norm = norm(target);

        if proj_norm < 1e-8 || tgt_norm < 1e-8 {
            continue;
        }

        let dot: f32 = projected
            .iter()
            .zip(target.iter())
            .map(|(a, b)| a * b)
            .sum();

        // Gradient: d(cos)/d(proj) = (target/|target| - proj*cos/(|proj|^2)) / |proj|
        let cos_sim = dot / (proj_norm * tgt_norm);

        for i in 0..output_dim {
            let d_proj = (target[i] / tgt_norm - projected[i] * cos_sim / proj_norm) / proj_norm;

            // Gradient w.r.t. weights: d_proj * embedding
            for j in 0..input_dim {
                let idx = i * input_dim + j;
                let grad = -d_proj * embedding[j]; // negative because loss = 1 - cos

                // SGD with momentum
                velocity[idx] = momentum * velocity[idx] + lr * grad;
                weights[idx] -= velocity[idx];
            }
        }
    }

    total_loss / embeddings.len() as f32
}

#[cfg(feature = "neural-bridge")]
fn compute_loss(weights: &[f32], embeddings: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
    let output_dim = HDC_DIMENSION;
    let mut total_loss = 0.0;

    for (embedding, target) in embeddings.iter().zip(targets.iter()) {
        let projected = project(weights, embedding, output_dim);
        let loss = 1.0 - cosine_similarity(&projected, target);
        total_loss += loss;
    }

    total_loss / embeddings.len() as f32
}

#[cfg(feature = "neural-bridge")]
fn project(weights: &[f32], embedding: &[f32], output_dim: usize) -> Vec<f32> {
    let input_dim = embedding.len();
    let mut result = vec![0.0f32; output_dim];

    for i in 0..output_dim {
        let mut sum = 0.0;
        for j in 0..input_dim {
            sum += weights[i * input_dim + j] * embedding[j];
        }
        result[i] = sum;
    }

    result
}

#[cfg(feature = "neural-bridge")]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = norm(a);
    let norm_b = norm(b);

    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(feature = "neural-bridge")]
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(feature = "neural-bridge")]
fn save_weights_npy(weights: &[f32], rows: usize, cols: usize, path: &PathBuf) -> Result<()> {
    use std::io::Write;

    // Create directory if needed
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // NumPy .npy format
    let mut file = std::fs::File::create(path)?;

    // Magic number
    file.write_all(&[0x93])?;
    file.write_all(b"NUMPY")?;

    // Version 1.0
    file.write_all(&[0x01, 0x00])?;

    // Header
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        rows, cols
    );

    // Pad header to multiple of 64 bytes (including magic + version + header_len)
    let header_len = header.len();
    let total_header = 10 + header_len; // 6 magic + 2 version + 2 header_len
    let padding = (64 - (total_header % 64)) % 64;
    let padded_header_len = header_len + padding;

    // Write header length (little endian u16)
    file.write_all(&(padded_header_len as u16).to_le_bytes())?;

    // Write header
    file.write_all(header.as_bytes())?;

    // Write padding (spaces + newline)
    for _ in 0..padding.saturating_sub(1) {
        file.write_all(b" ")?;
    }
    file.write_all(b"\n")?;

    // Write data (little endian f32)
    for &value in weights {
        file.write_all(&value.to_le_bytes())?;
    }

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
